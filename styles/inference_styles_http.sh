#!/usr/bin/env bash
curl 'http://localhost:8082/api/v2/tfproxy/' -H 'Proxy-Addr: localhost' -H 'content-type: multipart/form-data' -H 'accept: application/json, text/plain, */*' -F raw_input=true -F byte_images=@Screenshot_20180319_181842.png