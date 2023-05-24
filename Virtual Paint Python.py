# Proporciona un conjunto de soluciones de aprendizaje automático para tareas de percepción, como detección de manos 
import mediapipe as mp

#cv2 se utiliza para leer, escribir y mostrar imágenes y videos
import cv2

#numpy es una biblioteca de Python que proporciona una gran colección de objetos y funciones para trabajar con matrices
import numpy as np

#time proporciona varias funciones relacionadas con el tiempo
import time

#contsantes
ml = 150 #Representa la longitud máxima de dibujo permitida.
max_x, max_y = 250+ml, 50 #Definen las coordenadas máximas del área de dibujo.
curr_tool = "select tool" #Almacena la herramienta actual seleccionada.
time_init = True #Es una bandera booleana que indica si es la primera vez que se ejecuta el programa.
rad = 40 #Representa el radio del pincel utilizado para dibujar.
var_inits = False #Es una bandera booleana que indica si las variables ya han sido inicializadas.
thick = 4 #Representa el grosor del pincel utilizado para dibujar.
prevx, prevy = 0,0 #Almacena las coordenadas del punto anterior.

#get tools function

# Recibe un valor x y devuelve una cadena que representa la herramienta correspondiente según el valor de x.
# Dependiendo del rango en el que se encuentre x, se asignará una herramienta específica, como "line" (línea),
# "rectangle" (rectángulo), "draw" (dibujo), "circle" (círculo) o "erase" (borrar).
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"

# index_raised function

# Recibe dos valores yi y y9 que representan las coordenadas y del índice del dedo en dos momentos diferentes. 
# La función compara la diferencia entre las coordenadas y determina si el dedo está levantado o no. 
# Si la diferencia es mayor a 40, se considera que el dedo está levantado y la función devuelve True, de lo contrario, devuelve False.
def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False


# Asigna a la variable hands la solución de detección de manos proporcionada por Mediapipe
hands = mp.solutions.hands

# Crea una instancia de la clase Hands para detectar las manos en la imagen. 
# Se establecen los parámetros de confianza mínima de detección y seguimiento, así como el número máximo de manos a detectar.
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

# Asigna a la variable draw el módulo drawing_utils de Mediapipe, que proporciona funciones auxiliares para dibujar puntos y conexiones 
# en las manos detectadas.
draw = mp.solutions.drawing_utils


# drawing tools

# Carga una imagen llamada "tools.png" que contiene las herramientas de dibujo
tools = cv2.imread("tools.png")

# Se asegura de que los valores de los píxeles en la imagen sean de tipo uint8.
tools = tools.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')
'''
tools = np.zeros((max_y+5, max_x+5, 3), dtype="uint8")
cv2.rectangle(tools, (0,0), (max_x, max_y), (0,0,255), 2)
cv2.line(tools, (50,0), (50,50), (0,0,255), 2)
cv2.line(tools, (100,0), (100,50), (0,0,255), 2)
cv2.line(tools, (150,0), (150,50), (0,0,255), 2)
cv2.line(tools, (200,0), (200,50), (0,0,255), 2)
'''

cap = cv2.VideoCapture(0)
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y



			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]

	frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

	cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("paint app", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
 