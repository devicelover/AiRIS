
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from program import AiRIS

hostName = "localhost"
serverPort = 8080


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        print(self.path)
        if(self.path == '/detect_object'):
            airis = AiRIS()
            airis.objectDetection()


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    # try:
    webServer.serve_forever()
    # except KeyboardInterrupt:
    #     pass

    webServer.server_close()
    print("Server stopped.")