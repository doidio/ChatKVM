# ChatKVM

> 安全内网下 AI 自主看屏幕操控电脑

## 语音输入

安装语音依赖 `uv sync --extra voice`（PyTorch 保留 GPU 支持，识别使用 CPU），运行 `uv run --extra voice scripts/download-speech-models.py` 将 SenseVoiceSmall 和 FSMN-VAD 完整下载到本地，在 `config.toml` 中设置路径（相对于配置文件）：

```toml
[speech]
model = "models/SenseVoiceSmall"
vad_model = "models/fsmn-vad"
```

输入框中的麦克风可录制最长 120 秒的语音，CPU 离线识别后填回输入框，确认或修改文字后再发送。录音不写入对话历史；页面打开时后台预加载模型，跨对话复用，不阻塞文字聊天。删除 `[speech]` 可关闭语音输入。

浏览器需允许麦克风权限，并通过受信任的 HTTPS 或访问者设备自己的 `localhost` 打开页面；`http://100.64.x.x:8501` 不满足远程录音要求。

### 内网 HTTPS 部署

以下是已在 Ubuntu 主机 `100.64.0.1` 和 Windows Edge 验证的私有 CA 方案。证书中的域名和 IP、[Nginx 配置](scripts/nginx-chatkvm.conf)中的监听地址及域名都要按实际部署修改。私有 CA 不依赖公网续签，但每台访问设备都必须信任它。

1. 在管理员电脑的**仓库外**私有目录中用 OpenSSL 创建 CA 和网站证书。CA 私钥 `root-ca.key` 只留在管理员电脑，妥善备份，不要复制到服务器或提交到 Git。以下命令可在 Linux shell 或 Windows Git Bash 中运行：

   ```sh
   openssl req -x509 -newkey rsa:3072 -noenc -sha256 -days 3650 \
     -keyout root-ca.key -out root-ca.crt -subj '/CN=ChatKVM Private Root CA' \
     -addext 'basicConstraints=critical,CA:TRUE' \
     -addext 'keyUsage=critical,keyCertSign,cRLSign'
   openssl req -new -newkey rsa:2048 -noenc -sha256 \
     -keyout server.key -out server.csr -subj '/CN=app.chatkvm.com' \
     -addext 'subjectAltName=DNS:app.chatkvm.com,IP:100.64.0.1' \
     -addext 'extendedKeyUsage=serverAuth' \
     -addext 'keyUsage=critical,digitalSignature,keyEncipherment'
   openssl x509 -req -in server.csr -CA root-ca.crt -CAkey root-ca.key \
     -CAcreateserial -out server.crt -days 365 -sha256 -copy_extensions copy
   ```

2. 将 `server.crt` 和 `server.key` 通过 SSH 传到服务器上仅部署用户可访问的暂存目录，例如 `~/.cache/chatkvm-pki/`。在服务器的 ChatKVM 项目目录执行：

   ```sh
   sudo apt update
   sudo apt install nginx
   sudo install -d -m 700 /etc/ssl/chatkvm
   sudo install -m 644 ~/.cache/chatkvm-pki/server.crt /etc/ssl/chatkvm/server.crt
   sudo install -m 600 ~/.cache/chatkvm-pki/server.key /etc/ssl/chatkvm/server.key
   sudo install -m 644 scripts/nginx-chatkvm.conf /etc/nginx/sites-available/chatkvm
   sudo ln -s /etc/nginx/sites-available/chatkvm /etc/nginx/sites-enabled/chatkvm
   sudo install -d -m 755 /etc/systemd/system/nginx.service.d
   sudo install -m 644 scripts/nginx-chatkvm-override.conf /etc/systemd/system/nginx.service.d/chatkvm.conf
   sudo systemctl daemon-reload
   sudo nginx -t
   sudo systemctl enable --now nginx
   sudo systemctl reload nginx
   ```

   证书安装后删除服务器暂存目录中的私钥副本。若 Nginx 只服务 ChatKVM，可停用其默认 HTTP 站点。覆盖配置让 Nginx 在 xEdge 的 `tailscaled` 启动后运行，并在绑定虚拟网地址失败时重试；如果不使用该服务，按实际网络服务修改或省略覆盖配置。

3. 安装 [服务模板](scripts/chatkvm.service)并启动 ChatKVM：

   ```sh
   mkdir -p ~/.config/systemd/user
   install -m 644 scripts/chatkvm.service ~/.config/systemd/user/chatkvm.service
   systemctl --user daemon-reload
   systemctl --user enable --now chatkvm
   systemctl --user restart chatkvm
   ```

   模板假定项目位于 `~/chatkvm`；Streamlit 只监听 `127.0.0.1:8501`，外部设备通过 Nginx 的 `443` 端口访问，不再直接开放 HTTP `8501`。重启或无人登录后仍需运行时，可执行 `sudo loginctl enable-linger "$(whoami)"`。

4. 在访问设备上安装并信任 `root-ca.crt`，然后打开 `https://100.64.0.1/` 或已解析到该地址的 `https://app.chatkvm.com/`，点击输入框中的麦克风并允许权限。Windows 可将根证书导入“当前用户 → 受信任的根证书颁发机构”；iPhone 手动安装证书描述文件后，还需在“设置 → 通用 → 关于本机 → 证书信任设置”中开启完全信任。**不要**通过忽略浏览器证书警告来使用录音。

在已信任 CA 的 Windows Edge 中，`https://100.64.0.1/` 页面已验证可正常打开；iPhone 上的麦克风录音与识别仍需在该设备上实测。

域名访问还取决于客户端 DNS：联网时可使用指向虚拟网 IP 的公网 A 记录，完全离线时需自建内网 DNS 或配置 hosts。部分代理的假 IP DNS 也会影响解析。网站证书到期前要用保留的 CA 私钥重新签发并替换服务器证书；本方案不在服务器保存 CA 私钥，也不自动续签。

## 文本输入

Agent 只使用 `type_text` 向当前焦点键入文本，不调用目标电脑的命令行或剪贴板。默认采用美式键盘 ASCII 输入；在已验证的简体中文 Windows 目标设备上，可在对应的 `[kvmcp."设备名"]` 配置中设置 `text_input = "windows_gbk"`。该模式通过 HID 数字小键盘 Alt 编码输入 GBK 字符，换行和制表符使用物理按键，不依赖中文输入法。

工具逐字符处理文本，跳过无法编码的字符并返回其位置；发送途中失败则停止，已发送的部分可能留在目标窗口。键入前确认焦点，键入后核对内容。GBK 模式需在目标系统和实际软件中验证，不能保证支持所有 Unicode 字符或文本控件。
