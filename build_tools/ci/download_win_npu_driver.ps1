$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$session.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
$session.Cookies.Add((New-Object System.Net.Cookie("MRHSession", "29388b7d5a9f2f6c400f4fd2defd02bb", "/", "account.amd.com")))
$session.Cookies.Add((New-Object System.Net.Cookie("fonce_current_session", "1", "/", "account.amd.com")))
$session.Cookies.Add((New-Object System.Net.Cookie("fonce_current_day", "1,2024-08-15", "/", "account.amd.com")))
Invoke-WebRequest -UseBasicParsing -Uri "https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html" `
-Method "POST" `
-WebSession $session `
-Headers @{
    "authority"="account.amd.com"
    "method"="POST"
    "path"="/en/forms/downloads/ryzen-ai-software-platform-xef.html"
    "scheme"="https"
    "accept"="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    "accept-encoding"="gzip, deflate, br, zstd"
    "accept-language"="en-US,en;q=0.9"
    "cache-control"="max-age=0"
    "origin"="https://account.amd.com"
    "priority"="u=0, i"
    "referer"="https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html"
    "sec-ch-ua"="`"Not)A;Brand`";v=`"99`", `"Google Chrome`";v=`"127`", `"Chromium`";v=`"127`""
    "sec-ch-ua-mobile"="?0"
    "sec-ch-ua-platform"="`"Windows`""
    "sec-fetch-dest"="document"
    "sec-fetch-mode"="navigate"
    "sec-fetch-site"="same-origin"
    "sec-fetch-user"="?1"
    "upgrade-insecure-requests"="1"
} `
-ContentType "multipart/form-data; boundary=----WebKitFormBoundaryPSz5Z6dBITJcE9BB" `
-Body ([System.Text.Encoding]::UTF8.GetBytes("------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`":formstart`"$([char]13)$([char]10)$([char]13)$([char]10)/content/account/live-site/en/forms/downloads/ryzen-ai-software-platform-xef/jcr:content/root/container/container/container_1219627829$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"_charset_`"$([char]13)$([char]10)$([char]13)$([char]10)UTF-8$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`":redirect`"$([char]13)$([char]10)$([char]13)$([char]10)/en/forms/registration-messages/default/thank-you-sponsor.html$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Last_Name_Lang`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"First_Name_Lang`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Company_Lang`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"First_Name`"$([char]13)$([char]10)$([char]13)$([char]10)Max$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Last_Name`"$([char]13)$([char]10)$([char]13)$([char]10)Bob$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Email`"$([char]13)$([char]10)$([char]13)$([char]10)mxbob468@gmail.com$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Company`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Address_1`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Address_2`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Country`"$([char]13)$([char]10)$([char]13)$([char]10)US$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"State`"$([char]13)$([char]10)$([char]13)$([char]10)CA$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"City`"$([char]13)$([char]10)$([char]13)$([char]10)San Jose$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Zip_Code`"$([char]13)$([char]10)$([char]13)$([char]10)95124$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Phone`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"Job_Function`"$([char]13)$([char]10)$([char]13)$([char]10)CEO/CFO$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"filename`"$([char]13)$([char]10)$([char]13)$([char]10)NPU_RAI1.2_20240729.zip$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB$([char]13)$([char]10)Content-Disposition: form-data; name=`"agree-name`"$([char]13)$([char]10)$([char]13)$([char]10)Max Bob$([char]13)$([char]10)------WebKitFormBoundaryPSz5Z6dBITJcE9BB--$([char]13)$([char]10)")) `
-OutFile NPU_RAI1.2_20240729.zip