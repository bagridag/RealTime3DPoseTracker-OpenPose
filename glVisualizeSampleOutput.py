#!/usr/bin/python
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

name = "3d vis"
height = 400
width = 400
rotate = 0
beginx = 0.0
beginy = 0.0
rotx = 0.0
roty = 0.0
zoom = 0.0
action = ""


edgesPose = (
    (0,1),
    (1,2),
    (1,5),
    (2,3),
    (3,4),
    (5,6),
    (6,7),
    (1,8),
    (1,11),
    (8,9),
    (11,12),
    (9,10),
    (12,13),
    (0,15),
    (0,14),
    (14,16),
    (15,17)
    )

verticesPose= (
    (-0.176, 0.474, 1.677),
    (-0.190, 0.273, 1.678),
    (-0.343, 0.274, 1.678),
    (-0.438, 0.144, 1.483),
    (-0.382, 0.196, 1.318),
    (-0.0469, 0.2851, 1.751),
    (0.121, 0.170, 1.575),
    (0.117, 0.204, 1.428),
    (-0.244, -0.1498, 1.528),
    (-0.237, -0.544, 1.536),
    (-0.238, -0.721, 1.462),
    (-0.046, -0.1506, 1.565),
    (-0.0596, -0.5285, 1.574),
    (0.0, 0.0, 0.0),
    (-0.216, 0.5025, 1.698),
    (-0.156, 0.505, 1.706),
    (-0.278, 0.485, 1.768),
    (-0.125, 0.490, 1.786)
    )

verticesPoseStraightH= (
    (0.09290609,  0.3911047, 1.6204202),
    (0.07812399,  0.16361818, 1.593983),
    (-0.07601932,  0.17788392, 1.6440828),
    (-0.15906775, -0.04952209, 1.6227571),
    (-0.22482367,  0.08362344, 1.482427),
    (0.23522063,  0.1660381,  1.6624902),
    (0.3060596,  -0.04951069, 1.6223674),
    (0.37223142,  0.05219481, 1.485316),
    (-0.02705169, -0.25156453, 1.4625621),
    (-0.02648423, -0.6091409,  1.4319735),
    (0., 0., 0.),
    (0.17252973, -0.26898307, 1.4690416),
    (0.14653692, -0.6096989,  1.4148316),
    (0., 0., 0.),
    (0.05975833,  0.41830748, 1.6569456),
    (0.1275641,   0.41957837,  1.6619792),
    (0. , 0.38584808, 1.6892524),
    (0.16757679,  0.39048553,  1.7095553)
    )

edgesHand = (
    (0,1),
    (0,5),
    (0,9),
    (0,13),
    (0,17),
    (1,2),
    (2,3),
    (3,4),
    (5,6),
    (6,7),
    (7,8),
    (9,10),
    (10,11),
    (11,12),
    (13,14),
    (14,15),
    (15,16),
    (17,18),
    (18,19),
    (19,20)
    )

verticesLeftHand= (
    (-0.36178762, 0.20952047, 1.3172593),
    (-0.35002673,  0.20904374,  1.3142678),
    (-0.32848334,  0.21326905,  1.3254206),
    (-0.3237308,   0.22673282,  1.3111365),
    (-0.32959238,  0.23405835,  1.2913479),
    (-0.33671224,  0.25253403,  1.3003991),
    (-0.3254543,   0.25339818,  1.2986575),
    (-0.32195157,  0.25107393,  1.2990696),
    (-0.32378894,  0.24617663,  1.311372),
    (-0.35234106,  0.26243335,  1.3138337),
    (-0.33990347,  0.2465497,   1.2942328),
    (-0.3489648,   0.23304531,  1.3057513),
    (-0.35420978,  0.23090127,  1.3072784),
    (-0.36924505,  0.2639194,   1.3091518),
    (-0.35866064,  0.24508427,  1.2928241),
    (-0.36638403,  0.23216414,  1.3075824),
    (-0.37396476,  0.23115546,  1.3087289),
    (-0.3868784,   0.26032937,  1.3033006),
    (-0.37806267,  0.24406578,  1.2937545),
    (-0.3792922,   0.23045602,  1.297962),
    (-0.38277858,  0.2287042,   1.3016543)
    )



verticesLeftHandStraight= (
    (-0.2278364,   0.09140742,  1.4753048),
    (-0.21035226,  0.13028269,  1.4675423),
    (-0.18762484,  0.14927751,  1.4809908),
    (-0.17461264,  0.1718843,   1.4751743),
    (-0.15842775,  0.19227982,  1.4642677),
    (-0.22614539,  0.1792616,   1.491142),
    (-0.22949776,  0.2141979,   1.5040755),
    (-0.23165897,  0.23584558,  1.509098),
    (-0.23526779,  0.2576741,   1.5143485),
    (-0.24715796,  0.17397717,  1.4931258),
    (-0.25644553,  0.2146337,   1.507138),
    (-0.2614681,   0.23365235,  1.5039591),
    (-0.26522183,  0.25545043,  1.5094995),
    (-0.26100892,  0.16433895,  1.4933743),
    (-0.27114886,  0.20023298,  1.5036561),
    (-0.27763754,  0.2204361,   1.5086834),
    (-0.2808583,   0.24313104,  1.5110077),
    (-0.27648628,  0.15268645,  1.4874852),
    (-0.2911439,   0.17468607,  1.4992174),
    (-0.30196822,  0.18786082,  1.5047873),
    (-0.31172568,  0.20178664,  1.5048738)
    )




def display():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0,0,10,0,0,0,0,1,0)
    glRotatef(roty,0,1,0)
    glRotatef(rotx,1,0,0)
    glCallList(1)
    glutSwapBuffers()
    return

def mouse(button,state,x,y):
    global beginx,beginy,rotate
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        rotate = 1
        beginx = x
        beginy = y
    if button == GLUT_LEFT_BUTTON and state == GLUT_UP:
        rotate = 0
    return

def motion(x,y):
    global rotx,roty,beginx,beginy,rotate
    if rotate:
        rotx = rotx + (y - beginy)
        roty = roty + (x - beginx)
        beginx = x
        beginy = y
        glutPostRedisplay()

    return

def keyboard():
    return


def drawAxis():
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
#x-coordinate-red
    glVertex3f(-4.0, 0.0, 0.0)
    glVertex3f(4.0, 0.0, 0.0)

    glVertex3f(4.0, 0.0, 0.0)
    glVertex3f(3.0, 1.0, 0.0)

    glVertex3f(4.0, 0.0, 0.0)
    glVertex3f(3.0, -1.0, 0.0)
    glEnd()
    glFlush()


# y- coordinate-green
    glColor3f(0.0, 1.0, 0.0)

    glBegin(GL_LINES)
    glVertex3f(0.0, -4.0, 0.0)
    glVertex3f(0.0, 4.0, 0.0)

    glVertex3f(0.0, 4.0, 0.0)
    glVertex3f(1.0, 3.0, 0.0)

    glVertex3f(0.0, 4.0, 0.0)
    glVertex3f(-1.0, 3.0, 0.0)
    glEnd()
    glFlush()


#z-coordinate-blue
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, -4.0)
    glVertex3f(0.0, 0.0, 4.0)

    glVertex3f(0.0, 0.0, 4.0)
    glVertex3f(0.0, 1.0, 3.0)

    glVertex3f(0.0, 0.0, 4.0)
    glVertex3f(0.0, -1.0, 3.0)
    glEnd()
    glFlush()

glutInit(name)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(height,width)
glutCreateWindow(name)
glClearColor(0.0,0.0,0.0,1.0)



# setup display list
glNewList(1,GL_COMPILE)
glPushMatrix()
#glTranslatef(0.0,1.0,0.0) #move to where we want to put object

#glScalef(2.0,2.0,2.0)
drawAxis()


glLineWidth(2.0)
glColor3f(1.0, 0.9, 0.4)
glBegin(GL_LINES)
for edge in edgesPose:
    for vertex in edge:
        if (verticesPoseStraightH[vertex][0] == 0 and verticesPoseStraightH[vertex][1] == 0):
            continue
        glVertex3fv(verticesPoseStraightH[vertex])
glEnd()


glColor3f(0.6, 0.3, 0.5)
glBegin(GL_LINES)
for edge in edgesHand:
    for vertex in edge:
        # if (verticesPose[vertex][0] == 0 and verticesPose[vertex][1] == 0):
        # continue
        glVertex3fv(verticesLeftHandStraight[vertex])
glEnd()



#glutSolidSphere(1,20,20) # make radius 1 sphere of res 10x10
glPopMatrix()
glPushMatrix()
glTranslatef(0.0,-1.0,0.0) #move to where we want to put object
#glutSolidSphere(1,20,20) # make radius 1 sphere of res 10x10
glPopMatrix()
glEndList()

#setup lighting
glEnable(GL_CULL_FACE)
#glEnable(GL_DEPTH_TEST)
#glEnable(GL_LIGHTING)
#lightZeroPosition = [10.,4.,10.,1.]
#lightZeroColor = [0.8,1.0,0.8,1.0] # greenish
#glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
#glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
#glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
glEnable(GL_LIGHT0)

#setup cameras
glMatrixMode(GL_PROJECTION)
gluPerspective(20.,1.,1.,30.)
glMatrixMode(GL_MODELVIEW)
gluLookAt(0,0,10,0,0,0,0,1,0)
glPushMatrix()

#setup callbacks
glutDisplayFunc(display)
glutMouseFunc(mouse)
glutMotionFunc(motion)
glutKeyboardFunc(keyboard)

glutMainLoop()