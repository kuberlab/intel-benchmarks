wrk.method = "POST"
--wrk.headers["Content-Type"] = "application/json"
wrk.headers["Proxy-Port"] = "9000"
wrk.headers["Proxy-Addr"] = "10.5.2.1"