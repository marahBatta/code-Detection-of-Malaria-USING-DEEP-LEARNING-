#In this code, we defined a client to send a link to an image of an infected or healthy cell

from socket import*
cl_sock=socket()
cl_sock.connect(("127.0.0.1",5522))
while True :
     user_input=input("enter link of image:  ")
     cl_sock.send(user_input.encode('UTF-8'))
     if user_input=="exit":
         break
     res = cl_sock.recv(1024).decode()
     print(res) # wating
     res=cl_sock.recv(1024).decode()
     print(res)
print(" end ")
cl_sock.close()

#"Parasitised.png"
#"Uninfected.png"
