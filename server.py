import http.server
import socketserver
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus

# Start Streamlit in a separate process
streamlit_process = subprocess.Popen(
    [
        'streamlit', 'run', 
        '--server.port=8501',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        '--server.fileWatcherType=none',
        '--browser.serverAddress=localhost',
        'main.py'
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Give Streamlit some time to start
print("Waiting for Streamlit to start...")
time.sleep(5)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Proxy requests to Streamlit
        if self.path.startswith('/static'):
            # Serve static files with correct MIME types
            if self.path.endswith('.js'):
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.end_headers()
                with open(f'.{self.path}', 'rb') as f:
                    self.wfile.write(f.read())
                return
            elif self.path.endswith('.css'):
                self.send_header('Content-type', 'text/css')
        
        # Proxy API requests to Streamlit
        self.proxy_to_streamlit()
    
    def proxy_to_streamlit(self):
        import urllib.request
        import urllib.error
        
        try:
            # Forward the request to Streamlit
            url = f'http://localhost:8501{self.path}'
            req = urllib.request.Request(
                url,
                method=self.command,
                headers=self.headers
            )
            
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for header, value in response.getheaders():
                    # Skip hop-by-hop headers
                    if header.lower() in ('connection', 'transfer-encoding', 'content-encoding'):
                        continue
                    self.send_header(header, value)
                self.end_headers()
                self.copyfile(response, self.wfile)
                
        except urllib.error.HTTPError as e:
            self.send_error(e.code, str(e))
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        # Disable logging to keep the console clean
        pass

def start_server(port=8000):
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        print(f"Serving at port {port}")
        print(f"Proxying to Streamlit at http://localhost:8501")
        print(f"Open http://localhost:{port} in your browser")
        webbrowser.open_new_tab(f"http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            streamlit_process.terminate()
            httpd.shutdown()

if __name__ == "__main__":
    start_server(8000)
