from mnist import MNISTSession
import BaseHTTPServer
import json

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write("")

    def do_GET(self):
        self.send_response(403)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        jdata = json.dumps({"message": "Only POST requests are supported."})
        self.wfile.write(jdata)

    def do_POST(self):
        length = int(self.headers["content-length"])
        jdata = self.rfile.read(length)
        result = None
        try:
            data = json.loads(jdata)

            if type(data) == list:
                result = self.server.mnist.guess_number(data)

                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
                self.send_header("Content-type", "application/json")
                self.end_headers()

                self.wfile.write(json.dumps({"result": result}))
            elif "correct" in data:
                # TODO
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}))

        except ValueError:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            jdata = json.dumps({"message": "Failed to parse input data"})
            self.wfile.write(jdata)
            return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
        self.end_headers()
        self.wfile.write("")

if __name__ == "__main__":
    print("Creating and training MNIST session...")
    mnist = MNISTSession("/home/tensorflow/MNIST_data/")
    mnist.train()

    print("Starting HTTP server...")
    httpd = BaseHTTPServer.HTTPServer(("0.0.0.0", 8000), MyHandler)
    httpd.mnist = mnist

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    print("Stopping HTTP server...")
    httpd.server_close()