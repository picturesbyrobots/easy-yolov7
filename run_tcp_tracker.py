from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import selectors
import socket

import json

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

webcam = cv2.VideoCapture(0)
sel = selectors.DefaultSelector()
sel = selectors.DefaultSelector()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(5)  # Listen for incoming connections

sel.register(server_socket, selectors.EVENT_READ, data=None)

client_sockets = []

def send_json_to_clients(data):
    # Serialize the JSON data

    # Send the JSON data to all connected clients
    for client_socket in client_sockets:  # Skip the server socket
        try:
            json_data = json.dumps(data) + '\r\n' 
            client_socket.send(json_data.encode('utf-8'))
        except Exception as e:
            print(f"Error sending data to client: {e}")


if webcam.isOpened() == False:
	print('[!] error opening the webcam')

should_stop = False
try:
    while webcam.isOpened() and not should_stop:
        ret, frame = webcam.read()
        if ret == True:
            detections = yolov7.detect(frame, track=True)
            detected_frame = draw(frame, detections)
            send_json_to_clients(detections)
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
        
        else:
            break
        
        events = sel.select(timeout=0)
        for key, mask in events:
            if key.fileobj is server_socket :
                client_socket, client_address = key.fileobj.accept()
                print(f"New connection from {client_address}")
                client_socket.setblocking(False)
                sel.register(client_socket, selectors.EVENT_READ, data=None)
                client_sockets.append(client_socket) 
            else:
                client_socket = key.fileobj
                data = client_socket.recv(1024)
                if data:
                    #print(f"Received data from {client_socket.getpeername()}: {data.decode('utf-8')}")
                    
                    json_data = json.loads(data.decode('utf-8'))
                    print(json_data)
                    if json_data['command'] == 'stop':
                        should_stop = True
                        print('stopping')

                else:
                    print(f"Client {client_socket.getpeername()} has disconnected")
                    sel.unregister(client_socket)
                    client_socket.close()
                    client_sockets.remove(client_socket)
       
except KeyboardInterrupt:
    pass

server_socket.close()
webcam.release()
print('[+] webcam closed')
yolov7.unload()