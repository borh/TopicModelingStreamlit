import json
from typing import Any, Dict, List, Optional


def _widgets_list_from_at(at, name: str) -> List[Any]:
    """
    Return a list of widget objects for the given widget name from AppTest.

    The testing harness sometimes exposes widgets as attributes (lists) and
    sometimes via at.get(...) returning a callable or list; handle both shapes.
    """
    widgets = getattr(at, name, None)
    if isinstance(widgets, list):
        return widgets

    getter = getattr(at, "get", None)
    if callable(getter):
        try:
            got = getter(name)
            # Some harnesses return a helper callable; when it's a callable we can't call it here.
            if isinstance(got, list):
                return got
        except Exception:
            # Fall through to empty
            return []

    return []


def _find_widget(at, name: str, key: Optional[str]) -> Any:
    """
    Find the widget element by key (if provided) or return the first widget.
    If the named widget collection does not exist, try several common aliases
    (altair_chart, vega_lite_chart, plotly_chart) and return the first
    matching element found. Raises AssertionError with helpful debug info if none found.
    """
    # First try the exact requested name
    widgets = _widgets_list_from_at(at, name)
    if widgets:
        if key:
            for w in widgets:
                if getattr(w, "key", None) == key:
                    return w
            keys = [getattr(w, "key", None) for w in widgets]
            raise AssertionError(
                f"Widget list for '{name}' contains no element with key={key!r}. Available keys: {keys}"
            )
        return widgets[0]

    # If not found, try common alias names and search for the key across them
    candidate_names = ["altair_chart", "vega_lite_chart", "plotly_chart", "vega_chart"]
    tried = {}
    for cand in candidate_names:
        widgets_cand = _widgets_list_from_at(at, cand)
        if widgets_cand:
            tried[cand] = [getattr(w, "key", None) for w in widgets_cand]
            if key:
                for w in widgets_cand:
                    if getattr(w, "key", None) == key:
                        return w
            else:
                # no key requested, return first available widget
                return widgets_cand[0]

    # As a last resort, try calling at.get(name) which sometimes returns a list
    try:
        maybe = at.get(name)
        if isinstance(maybe, list) and maybe:
            if key:
                for w in maybe:
                    if getattr(w, "key", None) == key:
                        return w
            else:
                return maybe[0]
    except Exception:
        pass

    # Nothing found — provide helpful diagnostics
    available = [a for a in dir(at) if not a.startswith("_")]
    raise AssertionError(
        f"No widgets named '{name}' found on AppTest. Tried aliases: {list(tried.keys())}. "
        f"Found keys per alias: {tried}. Available attributes: {available}"
    )


def _parse_spec_from_obj(obj: Any) -> Dict:
    """
    Try to coerce a spec-like object into a Python dict.
    Accepts:
      - dict/list (returned as-is)
      - JSON string (parsed)
      - objects exposing to_json() or to_dict()
    Raises AssertionError on failure.
    """
    if obj is None:
        raise AssertionError("Chart spec object is None")

    if isinstance(obj, (dict, list)):
        return obj

    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Failed to parse JSON spec string: {e}; raw[:200]={obj[:200]!r}"
            )

    # try common methods
    if hasattr(obj, "to_json"):
        try:
            return json.loads(obj.to_json())
        except Exception as e:
            raise AssertionError(f"Failed to extract spec via to_json(): {e}")

    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception as e:
            raise AssertionError(f"Failed to extract spec via to_dict(): {e}")

    raise AssertionError(f"Unsupported chart spec type: {type(obj)}")


def _extract_spec_from_elem(elem: Any) -> Dict:
    """
    Try to extract a chart spec dict from a single widget element.
    Raises AssertionError if no plausible spec can be extracted.

    Enhanced behavior:
      - Prefer .proto-based extraction (Plotly .figure.spec, Altair .chart_spec / .spec)
      - Fallback to direct attributes (spec, chart_spec, figure, chart)
      - Finally try elem.value (works for UnknownElement which exposes the rendered value)
      - If we see an empty string we return a clear error message rather than raising JSON parse noise.
    """
    proto = getattr(elem, "proto", None)
    if proto is not None:
        # Plotly proto.figure.spec
        fig = getattr(proto, "figure", None)
        if fig is not None:
            spec = getattr(fig, "spec", None)
            if spec:
                return _parse_spec_from_obj(spec)
            if hasattr(fig, "to_json"):
                try:
                    raw = fig.to_json()
                    if raw and raw.strip():
                        return json.loads(raw)
                except Exception:
                    pass
            if hasattr(fig, "to_dict"):
                try:
                    return fig.to_dict()
                except Exception:
                    pass

        # Altair/Vega proto.chart_spec / proto.spec
        for attr in ("chart_spec", "spec", "chart"):
            spec_val = getattr(proto, attr, None)
            if spec_val:
                return _parse_spec_from_obj(spec_val)

    # Fallback to direct attributes on the element (some harnesses expose .spec / .chart directly)
    for attr in ("spec", "chart_spec", "figure", "chart"):
        val = getattr(elem, attr, None)
        if val:
            return _parse_spec_from_obj(val)

    # Final fallback: try elem.value (UnknownElement/value may hold the actual object or JSON)
    val = None
    try:
        val = getattr(elem, "value", None)
    except Exception:
        val = None

    if val is not None:
        # If it's an empty string, surface a clear assertion rather than raw JSON error
        if isinstance(val, str) and not val.strip():
            raise AssertionError("Empty chart spec string in elem.value (empty string)")
        try:
            return _parse_spec_from_obj(val)
        except AssertionError:
            # try parsing JSON string if it wasn't handled above
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    pass
            # If value is bytes, try decode
            if isinstance(val, (bytes, bytearray)):
                try:
                    return json.loads(val.decode("utf-8"))
                except Exception:
                    pass

    # As a last resort, include proto introspection in the error to aid debugging
    proto_repr = None
    try:
        proto_repr = repr(getattr(elem, "proto", None))
    except Exception:
        proto_repr = str(getattr(elem, "proto", None))
    raise AssertionError(
        f"Could not extract spec from element. elem repr: {repr(elem)[:300]}; proto repr (truncated): {proto_repr[:200]}"
    )


def list_all_chart_specs(
    at, widget_name: str = "plotly_chart"
) -> list[tuple[Optional[str], object]]:
    """
    Return a list of (key, spec_or_error) for all widgets under `widget_name`.
    Spec is a parsed dict/list when extraction succeeds, otherwise a string describing the error.
    """
    widgets = _widgets_list_from_at(at, widget_name)
    if not widgets:
        # last-resort attempt to call at.get(widget_name)
        try:
            maybe = at.get(widget_name)
            if isinstance(maybe, list):
                widgets = maybe
        except Exception:
            widgets = []

    out = []
    for w in widgets:
        key = getattr(w, "key", None)
        try:
            spec = _extract_spec_from_elem(w)
        except Exception as e:
            spec = f"<error extracting spec: {e}>"
        out.append((key, spec))
    return out


def _looks_like_umap_spec(spec: Dict) -> bool:
    """
    Heuristic to detect a 2D scatter (UMAP-like) chart spec.

    Returns True for common Altair/Vega-Lite and Plotly scatter representations.
    """
    if not isinstance(spec, dict):
        return False

    # Vega-Lite / Altair heuristics: look for layer/mark/encoding with x and y
    if "layer" in spec and isinstance(spec["layer"], list):
        for layer in spec["layer"]:
            if not isinstance(layer, dict):
                continue
            mark = layer.get("mark")
            enc = layer.get("encoding", {})
            if mark and (("point" in str(mark)) or ("circle" in str(mark))):
                if "x" in enc and "y" in enc:
                    return True
            if "x" in enc and "y" in enc:
                return True

    # top-level mark + encoding
    mark = spec.get("mark")
    encoding = spec.get("encoding", {})
    if mark and (("point" in str(mark)) or ("circle" in str(mark))):
        if "x" in encoding and "y" in encoding:
            return True
    if "x" in encoding and "y" in encoding:
        # could be a chart with both axes encoded
        return True

    # Plotly-style: look for data/traces with scatter and x/y
    # Plotly figure dicts often have "data": [ { "type": "scatter", "x":..., "y":... }, ... ]
    data = spec.get("data") or spec.get("traces") or spec.get("layers")
    if isinstance(data, list):
        for trace in data:
            if not isinstance(trace, dict):
                continue
            ttype = trace.get("type") or trace.get("chartType") or trace.get("mode")
            if isinstance(ttype, str) and "scatter" in ttype:
                if ("x" in trace and "y" in trace) or (
                    "xsrc" in trace and "ysrc" in trace
                ):
                    return True
            # some plotly specs embed x/y under 'x' and 'y' keys directly
            if "x" in trace and "y" in trace:
                return True

    return False


def get_chart_spec(
    at, widget_name: str = "plotly_chart", key: Optional[str] = None
) -> Dict:
    """
    Generic extractor to obtain a chart specification dict from a Streamlit AppTest.

    Enhanced fallback behavior:
      - First attempt to find widget by (widget_name, key) as before.
      - If not found, scan common chart collections (altair_chart, vega_lite_chart, plotly_chart, vega_chart)
        extracting specs from each element and:
          * prefer an element with matching key if present
          * prefer a UMAP-like 2D scatter spec if key not found (heuristic)
          * otherwise return the first spec found as a best-effort fallback
    """
    # Primary resolution: try exact lookup (may raise AssertionError)
    try:
        elem = _find_widget(at, widget_name, key)
        return _extract_spec_from_elem(elem)
    except AssertionError as primary_err:
        # Primary lookup failed — attempt content-based fallbacks across common widget collections
        candidate_names = [
            "altair_chart",
            "vega_lite_chart",
            "plotly_chart",
            "vega_chart",
        ]
        scanned = {}
        for cand in candidate_names:
            widgets = _widgets_list_from_at(at, cand)
            if not widgets:
                # Last-resort: try at.get(cand) which sometimes returns a list
                try:
                    maybe = at.get(cand)
                    if isinstance(maybe, list):
                        widgets = maybe
                except Exception:
                    widgets = []
            if not widgets:
                continue
            scanned[cand] = [getattr(w, "key", None) for w in widgets]
            specs = []
            for w in widgets:
                try:
                    s = _extract_spec_from_elem(w)
                    specs.append((w, s))
                except AssertionError:
                    continue
            if not specs:
                continue
            # 1) prefer exact key match if present
            if key:
                for w, s in specs:
                    if getattr(w, "key", None) == key:
                        print(
                            f"[get_chart_spec] Found widget by key under alias '{cand}': key={key!r}"
                        )
                        return s
            # 2) prefer UMAP-like scatter spec
            for w, s in specs:
                try:
                    if _looks_like_umap_spec(s):
                        print(
                            f"[get_chart_spec] Found UMAP-like spec under alias '{cand}' key={getattr(w, 'key', None)!r}"
                        )
                        return s
                except Exception:
                    continue
            # 3) fallback: return first spec found for this alias
            print(
                f"[get_chart_spec] Returning first spec from alias '{cand}' as fallback (keys={scanned[cand]})"
            )
            return specs[0][1]

        # Nothing found — raise informative error including what we scanned
        available = [a for a in dir(at) if not a.startswith("_")]
        raise AssertionError(
            f"Could not locate widget '{widget_name}' with key={key!r}. Primary error: {primary_err}. "
            f"Tried aliases: {list(scanned.keys())}. Found keys per alias: {scanned}. Available attributes: {available}"
        )
