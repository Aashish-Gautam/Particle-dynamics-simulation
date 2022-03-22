import pygame
import math
import numpy as np
from math import *
import time
pygame.init()
pygame.init()
background_colour = (255,255,160)
(width, height) = (1300, 700)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tutorial')
screen.fill(background_colour)
t=0.0
t1 = 0.001
def taninverse(x,y):
    if x == 0:
        if y>0:
            return pi/2
        if y<0:
            return -pi/2;
    if x>0 and y>=0:
        return atan(y/x)
    if x<0 and y<=0:
        return pi+atan(y/x) 
    if x<0 and y>=0:
        return pi - atan(abs(y)/abs(x))
    if x>0 and y<=0:
        return 2*pi-atan(abs(y)/abs(x))

class Particle:
    def __init__(self,(x,y),radius,speed,angle):
        
        self.x = x
        self.y = y
        self.rho = 1
        self.radius = radius
        self.colour = colour
        self.thickness = 0
        self.speed = speed
        self.angle = angle
        self.vx = self.speed*cos(self.angle)
        self.vy = self.speed*sin(self.angle)
        self.m = self.rho*pi*(self.radius**2)
 

    def make(self):
        pygame.draw.circle(screen,self.colour,(int(self.x),int(self.y)),int(self.radius),self.thickness)
     
    def move(self):

        # vxo=100
        # vyo=0

        # self.vx+=(0.01/2.0)*(vxo-self.vx)
        # self.vy+=((0.01/2.0))*(vyo-self.vy)
        # self.speed = hypot(self.vx,self.vy)
        # self.angle = taninverse(self.vx,self.vy)

        self.x += math.cos(self.angle)*self.speed/100
        self.y += math.sin(self.angle)*self.speed/100

        

    def bounce(self,e1,u1):
                
        if self.x > width - self.radius:
            u1o = u1*2*abs(self.vx)*(self.radius**2)
            if u1 == -1:
                eu1=0
            else:
                eu1 = 1-(2/pi)*atan(u1o)
            self.vx = -e1*self.speed*(cos(self.angle))
            self.vy = eu1*self.speed*sin(self.angle)
            self.speed = hypot(self.vx,self.vy)
            self.angle=taninverse(self.vx,self.vy)
            self.x = (width-self.radius)-(self.x-(width-self.radius))
            return True
            #    self.angle = pi - self.angle
        elif self.x < self.radius:
            u1o = u1*2*abs(self.vx)*(self.radius**2)
            if u1 == -1:
                eu1=0
            else:
                eu1 = 1-(2/pi)*atan(u1o)
            self.vx = -e1*self.speed*(cos(self.angle))
            self.vy = eu1*self.speed*sin(self.angle)
            self.speed = hypot(self.vx,self.vy)
            self.angle=taninverse(self.vx,self.vy)            
            self.x = 2*self.radius-self.x
            return True
            #    self.angle = pi - self.angle
        elif self.y > height - self.radius:
            u1o = u1*2*abs(self.vy)*(self.radius**2)
            if u1 == -1:
                eu1=0
            else:
                eu1 = 1-(2/pi)*atan(u1o)
            self.vx = eu1*self.speed*(cos(self.angle))
            self.vy = -e1*self.speed*(sin(self.angle))
            self.speed = hypot(self.vx,self.vy)
            self.angle=taninverse(self.vx,self.vy)       
            self.y = 2 * (height - self.radius) - self.y
            return True
            #    self.angle = - self.angle
        elif self.y < self.radius:
            u1o = u1*2*abs(self.vy)*(self.radius**2)
            if u1 == -1:
                eu1=0
            else:
                eu1 = 1-(2/pi)*atan(u1o)
            self.vx = eu1*self.speed*(cos(self.angle))
            self.vy = -e1*self.speed*(sin(self.angle))
            self.speed = hypot(self.vx,self.vy)
            self.angle=taninverse(self.vx,self.vy)    
            self.y = 2 * self.radius - self.y
            return True
            #    self.angle = - self.angle
        return False

    def collide(A,B,e2,u2):
        #eu2 = 1-(2/pi)*atan(u2)
        r = A.radius/B.radius
        if B.x!=A.x:
            m =  (B.y-A.y)/(B.x-A.x)
            a = taninverse(B.x-A.x,B.y-A.y)
            c = ((A.radius+B.radius)/((1+m**2)**(0.5))) 
            if m==0:
                d = 0 
            else:
                d = ((A.radius+B.radius)/((1+(1/m)**2)**(0.5)))
        elif B.x==A.x:
            a=pi/2
            c=0
            d=A.radius+B.radius

        if math.hypot((A.x-B.x),(A.y-B.y)) < A.radius+B.radius:
            Ax=0
            Bx=0
            Ay=0
            By=0
            if B.x>=A.x:
                Ax=(r**2*A.x+B.x-c)/(1+r**2)
                Bx=Ax+c
                if B.y>=A.y:
                    Ay=(r**2*A.y+B.y-d)/(1+r**2)
                    By=Ay+d
                elif A.y>B.y:
                    Ay=(r**2*A.y+B.y+d)/(1+r**2)
                    By=Ay-d 
            elif A.x>B.x:
                Ax=(r**2*A.x+B.x+c)/(1+r**2)
                Bx=Ax-c 
                if B.y>=A.y:
                    Ay=(r**2*A.y+B.y-d)/(1+r**2)
                    By=Ay+d
                elif A.y>B.y:
                    Ay=(r**2*A.y+B.y+d)/(1+r**2)
                    By=Ay-d  

            A.x=Ax
            A.y=Ay
            B.x=Bx
            B.y=By
            
            Aui= A.speed*math.cos(A.angle-a)
            Bui= B.speed*math.cos(B.angle-a)
            Auj= A.speed*math.sin(A.angle-a)
            Buj= B.speed*math.sin(B.angle-a)

            Avi=0
            Avj=0
            Bvi=0
            Bvj=0

            Avi= ((A.radius**2-B.m*e2)*Aui+(1+e2)*B.m*Bui)/(A.radius**2+B.radius**2)
            Bvi = Avi + e2*(Aui - Bui)

            u2o = u2*(A.radius**2)*abs(Avi-Aui)
            if u2==-1:
                eu2 = 0
            else:
                eu2 = 1-(2/pi)*atan(u2o)

            Avj= ((A.radius**2+B.m*eu2)*Auj + (1-eu2)*B.m*Buj)/(A.radius**2+B.radius**2)
            Bvj = Avj - eu2*(Auj - Buj)


            A.speed=math.hypot(Avi,Avj)
            B.speed=math.hypot(Bvi,Bvj)
            A.angle=(taninverse(Avi,Avj) + a)
            B.angle=(taninverse(Bvi,Bvj) + a)
            A.vx = A.speed*cos(A.angle)
            A.vy = A.speed*sin(A.angle)
            B.vx = B.speed*cos(B.angle)
            B.vy = B.speed*sin(B.angle)

class Rod:
    def __init__(self,(x,y),radius,speed,angle,w,alpha,length):
        self.x = x
        self.y = y
        self.rho = 1
        self.radius = radius
        self.colour = colour
        self.thickness = 0
        self.speed = speed
        self.angle = angle  #direction of speed
        self.length = length  #number of spheres to make a rod
        self.alpha = alpha  #initial angle of rod wrt horizontal
        self.w = w  #angular speed
        self.m1 = self.rho*((pi*(self.radius**2))/2)
        self.m2 = self.rho*self.length*2*self.radius
        self.m = 2*self.m1+self.m2

        
        a_rod=[]
        for i in range(self.length):
            x = self.x + self.R(i)*cos(self.alpha)
            y = self.y + self.R(i)*sin(self.alpha)
            a_rod.append(Particle((x,y),self.radius,self.speed,self.angle))
        I = ((1.0/12.0)*self.m2*(self.length**2) + 2*(self.m1*(self.radius**2)*(1.0-((4.0/(3.0*pi))**2)) + self.m1*((self.length/2 + (4.0*self.radius)/(3.0*pi))**2)))
        self.a_rod = a_rod
        self.I = I # moment of inertia

    def R(self,i): #distance of a ball from centre
        return -((self.length/2.0)-i)
    def make(self):  
        for sphere in self.a_rod:
            sphere.make()

    def move(self):  
    
        for i,sphere in enumerate(self.a_rod):
            if t==0:
                delta_x1 = 0.0
                delta_y1 = 0.0
                delta_vx1 = 0.0
                delta_vy1 = 0.0
            else:
                delta_x1 = (self.x + self.R(i)*cos(self.alpha))-(self.x + self.R(i)*cos(self.alpha-self.w*t1))
                delta_y1 = (self.y + self.R(i)*sin(self.alpha))-(self.y + self.R(i)*sin(self.alpha-self.w*t1))
                delta_vx1 = self.w*(-self.R(i)*sin(self.alpha)) - self.w*(-self.R(i)*sin(self.alpha-self.w*t1))
                delta_vy1 = self.w*(self.R(i)*cos(self.alpha)) - self.w*(self.R(i)*cos(self.alpha-self.w*t1))
            sphere.x += delta_x1 + 0.0
            sphere.y += delta_y1 + 0.0
            sphere.vx += delta_vx1 + 0.0
            sphere.vy += delta_vy1 + 0.0
            
            sphere.move()
            
        self.alpha += self.w*t1
        if self.alpha>2*pi:
            self.alpha=self.alpha%(2*pi)
        if self.alpha<(-2*pi):
            self.alpha=(2*pi+self.alpha)%(2*pi) - 2*pi


        
    def rigidity(self):
        for i,A in enumerate(self.a_rod):
            for B in self.a_rod[i+1:]:
                Particle.collide(A,B,0,-1)

    def collide(self,B,e3,u3):
        min_distance=self.radius+B.radius
        i = 0
        collision = 0
        for my_i,A in enumerate(self.a_rod):
            distance = abs(math.hypot((A.x-B.x),(A.y-B.y)))

            if distance < A.radius+B.radius:
                collision = 1
                if distance < min_distance:
                    min_distance = distance
                    i = my_i
                    
        if collision == 1:
            r = A.radius/B.radius
            if B.x!=A.x:
                m =  (B.y-A.y)/(B.x-A.x)
                a = taninverse(B.x-A.x,B.y-A.y)
                c = ((A.radius+B.radius)/((1+m**2)**(0.5))) 
                if m==0:
                    d = 0 
                else:
                    d = ((A.radius+B.radius)/((1+(1/m)**2)**(0.5)))
            elif B.x==A.x:
                a=pi/2
                c=0
                d=A.radius+B.radius
            Ax=0
            Bx=0
            Ay=0
            By=0
            if B.x>=A.x:
                Ax=(r**2*A.x+B.x-c)/(1+r**2)
                Bx=Ax+c
                if B.y>=A.y:
                    Ay=(r**2*A.y+B.y-d)/(1+r**2)
                    By=Ay+d
                elif A.y>B.y:
                    Ay=(r**2*A.y+B.y+d)/(1+r**2)
                    By=Ay-d 
            elif A.x>B.x:
                Ax=(r**2*A.x+B.x+c)/(1+r**2)
                Bx=Ax-c 
                if B.y>=A.y:
                    Ay=(r**2*A.y+B.y-d)/(1+r**2)
                    By=Ay+d
                elif A.y>B.y:
                    Ay=(r**2*A.y+B.y+d)/(1+r**2)
                    By=Ay-d  
            
            delta_Ax = Ax - A.x
            delta_Ay = Ay - A.y
            for sphere in self.a_rod:
                sphere.x += delta_Ax
                sphere.y += delta_Ay
            B.x=Bx 
            B.y=By
            

            KE1=0
            for g,h in enumerate(self.a_rod):
                KE1+=0.5*(self.radius**2)*((h.vx**2)+(h.vy**2))
            KE1+=0.5*(B.m*((B.vx**2)+(B.vy**2)))

            Aspeed=math.hypot(A.vx,A.vy)
            Bspeed=math.hypot(B.vx,B.vy)
            #After Collision
            beta = taninverse((A.x-B.x),(A.y-B.y)) # Collision angle wrt horizontal
            alpha = self.alpha
            u1x = B.vx
            u1y = B.vy

            Ucmx = self.speed*cos(self.angle)
            Ucmy = self.speed*sin(self.angle)

            u2x = A.vx
            u2y = A.vy

            u1i = u1x*cos(beta) + u1y*sin(beta)
            u1j = -u1x*sin(beta) + u1y*cos(beta)
            u2i = u2x*cos(beta) + u2y*sin(beta)
            u2j = -u2x*sin(beta) + u2y*cos(beta)

            if abs(u1i)<abs(u2i):
                beta=beta-pi

            u1i = u1x*cos(beta) + u1y*sin(beta)
            u1j = -u1x*sin(beta) + u1y*cos(beta)
            u2i = u2x*cos(beta) + u2y*sin(beta)
            u2j = -u2x*sin(beta) + u2y*cos(beta)

            rows = 4
            columns = 4
            v = [[0 for p in range(columns)] for j in range(rows)]
            v[0][0] = self.m*cos(beta)
            v[0][1] = self.m*sin(beta)
            v[0][2] = 0.0
            v[0][3] = B.m

            v[1][0] = 0.0
            v[1][1] = 0.0
            v[1][2] = self.I
            v[1][3] = B.m*self.R(i)*(sin(beta)*cos(alpha)-cos(beta)*sin(alpha))

            v[2][0] = cos(beta)
            v[2][1] = sin(beta)
            v[2][2] = self.R(i)*sin(beta-alpha)
            v[2][3] = -1.0

            v[3][0] = -sin(beta)
            v[3][1] = cos(beta)
            v[3][2] = self.R(i)*cos(beta-alpha)
            v[3][3] = 0.0


            k=[[0] for j in range(rows)]

            k[0][0] = B.m*u1i + self.m*(Ucmx*cos(beta)+Ucmy*sin(beta))
            k[1][0] = B.m*self.R(i)*(u1y*cos(alpha)-u1x*sin(alpha))-(B.m*self.R(i))*(cos(beta)*cos(alpha)+sin(beta)*sin(alpha))*u1j + self.I*self.w
            k[2][0] = e3*(u1i-u2i)
            k[3][0] = u2j

            Ans = [[0] for j in range(rows)]
            v=np.asarray(v, dtype='float')
            k=np.asarray(k, dtype='float')
            k=np.reshape(np.asarray(k, dtype='float'),(4,1))
            Ans = np.linalg.solve(v,k)

            Vcmx = Ans[0]
            Vcmy = Ans[1]
            wnew = Ans[2]

            v1i = Ans[3]
            v1j = u1j
            v2i = v1i + e3*(u1i-u2i)
            v2j = u2j

            v1x = -v1j*sin(beta) + v1i*cos(beta)
            v1y = v1i*sin(beta) + v1j*cos(beta)
            
            #print("before--" , B.m*self.R(i)*(sin(beta)*cos(alpha)-cos(beta)*sin(alpha))*u1i )
            #print("after--" , B.m*self.R(i)*(sin(beta)*cos(alpha)-cos(beta)*sin(alpha))*v1i)
            #print("alpha",self.alpha,"beta",beta,"beta-alpha",beta-alpha,"sin",sin(beta-alpha))
            #print(i)
            #print(self.R(i))
            #print(self.x,A.x,self.x-A.x)
            #print(B.radius)
            #print((sin(beta)*cos(alpha)-cos(beta)*sin(alpha)))
            #Solving for tangential part
            #print("u1i",u1i,"v1i",v1i)
            #print("u2i",u2i,"v2i",v2i)

            u3o = u3*B.m*abs(v1i-u1i)
            if u3==-1:
                eu3 = 0
            else:
                eu3 = 1-(2/pi)*atan(u3o)
            
            rows = 4
            columns = 4
            v = [[0 for p in range(columns)] for q in range(rows)]
            v[0][0] = -self.m*sin(beta)
            v[0][1] = self.m*cos(beta)
            v[0][2] = 0.0
            v[0][3] = B.m

            v[1][0] = 0.0
            v[1][1] = 0.0
            v[1][2] = self.I
            v[1][3] = B.m*self.R(i)*(cos(beta)*cos(alpha)+sin(beta)*sin(alpha))
        
            v[2][0] = cos(beta)
            v[2][1] = sin(beta)
            v[2][2] = self.R(i)*sin(beta-alpha)
            v[2][3] = 0.0

            v[3][0] = -sin(beta)
            v[3][1] = cos(beta)
            v[3][2] = self.R(i)*cos(beta-alpha)
            v[3][3] = -1.0

            k=[[0] for j in range(rows)]

            k[0][0] = B.m*u1j + self.m*(-Ucmx*sin(beta)+Ucmy*cos(beta))
            k[1][0] = B.m*self.R(i)*(u1y*cos(alpha)-u1x*sin(alpha))-B.m*self.R(i)*(sin(beta)*cos(alpha)-cos(beta)*sin(alpha))*u1i + self.I*self.w
            k[2][0] = u2i
            k[3][0] = -eu3*(u1j-u2j)

            Answer = [[0]for j in range(rows)]
            
            v=np.asarray(v, dtype='float')
            k=np.reshape(np.asarray(k, dtype='float'),(4,1))
            Answer = np.linalg.solve(v,k)
            

            Vfcmx = Vcmx + Answer[0] - Ucmx
            Vfcmy = Vcmy + Answer[1] - Ucmy
            wfinal = wnew + Answer[2] - self.w

            vf1i = v1i
            vf1j = Answer[3]
            vf2i = v2i
            vf2j = vf1j - (u1j-u2j)*eu3

            print("Ucmx",Ucmx,"Ucmy",Ucmy)
            print("Vcmx",Vcmx,"Vcmy",Vcmy)
            self.speed = math.hypot(Vfcmx,Vfcmy)

            #print(self.speed)
            self.angle = taninverse(Vfcmx,Vfcmy)
            #print(self.angle)
            #print("w_initial",self.w)
            self.w = wfinal
            #print("w_final",self.w)

            for t,S in enumerate(self.a_rod):
                S.speed = self.speed
                S.angle = self.angle
                S.vx = self.speed*cos(self.angle) - self.R(t)*self.w*sin(alpha)
                S.vy = self.speed*sin(self.angle) + self.R(t)*self.w*cos(alpha)
            
            

            vf1x = -vf1j*sin(beta) + vf1i*cos(beta)
            vf1y = vf1i*sin(beta) + vf1j*cos(beta)
            B.vx = vf1x
            B.vy = vf1y
            B.speed = math.hypot(vf1x,vf1y)
            B.angle = taninverse(vf1x,vf1y)

            

            KE2=0
            for g,h in enumerate(self.a_rod):
                KE2+=0.5*(self.radius**2)*((h.vx**2)+(h.vy**2))
                KE2+=0.5*(B.m*((B.vx**2)+(B.vy**2)))

            #print(KE1,KE2)
            KE1=0
            KE2=0
            
def collide2(self,B,e4,u4):
    for i,A in enumerate(self.a_rod):
        for q,B in enumerate(B.a_rod):
            r = A.radius/B.radius
            if B.x!=A.x:
                m =  (B.y-A.y)/(B.x-A.x)
                a = taninverse(B.x-A.x,B.y-A.y)
                c = ((A.radius+B.radius)/((1+m**2)**(0.5))) 
                if m==0:
                    d = 0 
                else:
                    d = ((A.radius+B.radius)/((1+(1/m)**2)**(0.5)))
            elif B.x==A.x:
                a=pi/2
                c=0
                d=A.radius+B.radius

            if math.hypot((A.x-B.x),(A.y-B.y)) < A.radius+B.radius:
                Ax=0
                Bx=0
                Ay=0
                By=0
                if B.x>=A.x:
                    Ax=(r**2*A.x+B.x-c)/(1+r**2)
                    Bx=Ax+c
                    if B.y>=A.y:
                        Ay=(r**2*A.y+B.y-d)/(1+r**2)
                        By=Ay+d
                    elif A.y>B.y:
                        Ay=(r**2*A.y+B.y+d)/(1+r**2)
                        By=Ay-d 
                elif A.x>B.x:
                    Ax=(r**2*A.x+B.x+c)/(1+r**2)
                    Bx=Ax-c 
                    if B.y>=A.y:
                        Ay=(r**2*A.y+B.y-d)/(1+r**2)
                        By=Ay+d
                    elif A.y>B.y:
                        Ay=(r**2*A.y+B.y+d)/(1+r**2)
                        By=Ay-d  
                
                delta_Ax = Ax - A.x
                delta_Ay = Ay - A.y
                for sphere in self.a_rod:
                    sphere.x += delta_Ax
                    sphere.y += delta_Ay

                delta_Bx = Bx - B.x
                delta_By = By - B.y
                for sphere in self.a_rod:
                    sphere.x += delta_Bx
                    sphere.y += delta_By

                #After Collision
                beta = taninverse((A.x-B.x),(A.y-B.y)) # Collision angle wrt horizontal
                alpha1 = self.alpha
                alpha2 = B.alpha

                U1cmx = B.speed*cos(B.angle)
                U1cmy = B.speed*sin(B.angle)
                u1x = B.vx
                u1y = B.vy

                U2cmx = self.speed*cos(self.angle)
                U2cmy = self.speed*sin(self.angle)
                u2x = A.vx
                u2y = A.vy

                u1i = u1x*cos(beta) + u1y*sin(beta)
                u1j = -u1x*sin(beta) + u1y*cos(beta)
                u2i = u2x*cos(beta) + u2y*sin(beta)
                u2j = -u2x*sin(beta) + u2y*cos(beta)

                if abs(u1i)<abs(u2i):
                    beta=beta-pi

                u1i = u1x*cos(beta) + u1y*sin(beta)
                u1j = -u1x*sin(beta) + u1y*cos(beta)
                u2i = u2x*cos(beta) + u2y*sin(beta)
                u2j = -u2x*sin(beta) + u2y*cos(beta)

                rows = 6
                columns = 6
                v = [[0 for p in range(columns)] for j in range(rows)]
                v[0][0] = B.length*B.m*cos(beta)
                v[0][1] = B.length*B.m*sin(beta)
                v[0][2] = 0.0
                v[0][3] = self.length*self.m*cos(beta)
                v[0][4] = self.length*self.m*sin(beta)
                v[0][5] = 0.0






                                 

    def bounce(self):
        for i,sphere in enumerate(self.a_rod):
            bounced = sphere.bounce(e1,u1)
            if bounced == True:
                self.ensureRigidity(i)
                return
            
    def ensureRigidity(self,i):
        for sphere in self.a_rod:
            sphere.speed = self.a_rod[i].speed
            sphere.angle = self.a_rod[i].angle

import random
# Creation of Particles
no_of_particles = 1
my_particles = []
for n in range(no_of_particles):

    radius = 20 #random.randint(10,35)
    x = 650 #random.randint(radius, width-radius)
    y = 0 #random.randint(radius, height-radius)
    speed = 300 #random.randint(500,760)
    angle = random.uniform(-pi,pi)
    colour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    my_particles.append(Particle((x, y),radius,speed,angle))

# Creation of Rods
no_of_rods = 1
my_rods = []
for n in range(no_of_rods):
    radius = 20 #random.randint(10,20)
    x = 650.0 #random.randint(radius, width-radius)
    y = 350.0 #random.randint(radius, height-radius)
    speed = 0.0 #random.randint(150,500)
    angle = pi/2 #random.uniform(-pi,pi)
    length = 500  #random.randint(5,8)
    colour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    w = 0.1
    alpha = 0.0
    my_rods.append(Rod((x,y),radius,speed,angle,w,alpha,length))

e1 = 1.0 #coeff. of resitution btw wall & spheres
u1 = 0.0 #coeff. of friction btw wall & spheres
e2 = 1.0 #coeff. of resitution btw spheres
u2 = 0.0 #coeff. of friction btw spheres
e3 = 1.0 #coeff. of resitution btw rod and spheres
u3 = 0.0 #coeff. of friction btw rod and spheres

while(True):
    screen.fill(background_colour)
    for i,particle in enumerate(my_particles):

        particle.make()
        particle.bounce(e1,u1)
        particle.move()
        
        for particle2 in my_particles[i+1:]:
            Particle.collide(particle,particle2,e2,u2)

    for i,rod in enumerate(my_rods):
        rod.make()
        #rod.rigidity()
        #rod.bounce()
        rod.move()


        for particle in my_particles:
           rod.collide(particle,e3,u3)

        # for rod2 in my_rods[i+1:]:
        #     for particle in rod2.a_rod:
        #         rod.collide(particle,e2,u2)
       
    pygame.display.flip()
    time.sleep(t1)
    t+=t1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        pygame.display.update()

