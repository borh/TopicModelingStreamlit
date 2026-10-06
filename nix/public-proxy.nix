{
  name,
  upstream,
  publicUpstream,
}:
''
  redir /${name} /${name}/
  route /${name}/* {
    header {
      X-Content-Type-Options nosniff
      Referrer-Policy same-origin
      Content-Security-Policy "frame-ancestors 'self'"
      Strict-Transport-Security "max-age=31536000"
    }
    @${name}Uploads {
      path /${name}/_stcore/upload_file/*
      not remote_ip 133.1.0.0/16
    }
    respond @${name}Uploads "Uploads require a university address." 403
    @${name}Campus remote_ip 133.1.0.0/16
    handle @${name}Campus {
      reverse_proxy ${upstream} {
        header_up X-Topic-Client-IP {remote_host}
        lb_policy cookie
      }
    }
    handle {
      reverse_proxy ${publicUpstream} {
        header_up X-Topic-Client-IP {remote_host}
        lb_policy cookie
      }
    }
  }
''
