function read_txt_file(path)
    local file, errorMessage = io.open(path, "r")
    if not file then 
        error("Could not read the file:" .. errorMessage .. "\n")
    end

    local content = file:read "*all"
    file:close()
    return content
end

local Boundary = "----WebKitFormBoundaryePkpFF7tjBAqx29L"
local BodyBoundary = "--" .. Boundary
local LastBoundary = "--" .. Boundary .. "--"

local CRLF = "\r\n"

local Filename = "speech.tf"
local FileBody = read_txt_file(Filename)

local DispositionRaw = "Content-Disposition: form-data; name=\"raw_input\""
local Raw = "yes"
local ContentDisposition = "Content-Disposition: form-data; name=\"bytes_audio\"; filename=\"" .. Filename .. "\""

wrk.method = "POST"
wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. Boundary
wrk.headers["Proxy-Port"] = "9000"
wrk.headers["Proxy-Addr"] = "10.5.2.2"
wrk.body = BodyBoundary .. CRLF .. ContentDisposition .. CRLF .. CRLF .. FileBody .. CRLF .. BodyBoundary .. CRLF .. DispositionRaw.. CRLF .. CRLF .. Raw .. CRLF ..  LastBoundary
