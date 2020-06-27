import cv2
import numpy as np
import imutils
import csv
import boto3




sudoku_img = cv2.imread("test25.png")
orig = cv2.resize(sudoku_img.copy(),(550,550))
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(gray.copy(), (5, 5), 1)
proc = cv2.adaptiveThreshold(imgBlur, 255,1,1,11, 3)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# edged = cv2.Canny(gray, 30, 200)
contours,h = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
test = cv2.rectangle(orig.copy(),[screenCnt],(0,255,0),3)
# cv2.imshow("Game Boy Screen", test)
# cv2.waitKey(0)
cv2.drawContours(sudoku_img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Game Boy Screen", sudoku_img)
cv2.waitKey(0)

pts = screenCnt.reshape(4, 2)
rect = np.zeros((4, 2), dtype = "float32")
# the top-left point has the smallest sum whereas the
# bottom-right has the largest sum
s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
# compute the difference between the points -- the top-right
# will have the minumum difference and the bottom-left will
# have the maximum difference
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# ...and now for the height of our new image
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# take the maximum of the width and height values to reach
# our final dimensions
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))
# construct our destination points which will be used to
# map the screen to a top-down, "birds eye" view
dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
# calculate the perspective transform matrix and warp
# the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

#warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
#warp = exposure.rescale_intensity(warp, out_range = (0, 12))
# the pokemon we want to identify will be in the top-right
# corner of the warped image -- let's crop this region out
# (h, w) = warp.shape
# (dX, dY) = (int(w * 0.4), int(h * 0.45))
# crop = warp[10:dY, w - dX:w - 10]
# save the cropped image to file
cv2.imshow("Game Boy Screen", warp)
cv2.waitKey(0)
cv2.imwrite("cropped.jpg", warp)
digits = []
minTop = np.float64(1)
maxTop = np.float64(0)
minLeft = np.float64(1)
maxLeft = np.float64(0)
photo = "cropped.jpg"
with open("new_user_credentials.csv","r") as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

client = boto3.client("rekognition",aws_access_key_id = access_key_id,aws_secret_access_key = secret_access_key,region_name='us-east-1')

with open(photo,"rb") as source_image:
    source_bytes = source_image.read()
response = client.detect_text(Image = {"Bytes":source_bytes})


for detect in response["TextDetections"]:
    #print()
    if detect["Type"]=="WORD":
        numbers = []
        detected_text = detect["DetectedText"]
        for i in range(len(detected_text)):
            if detected_text[i] == str(detected_text[i:i+1]):
                numbers.append(int(detected_text[i]))
        #print(numbers)


        if len(numbers)==0:
            continue
        if detect["Geometry"]["BoundingBox"]["Left"] < minLeft:
            minLeft = detect["Geometry"]["BoundingBox"]["Left"]

        if detect["Geometry"]["BoundingBox"]["Left"] + detect["Geometry"]["BoundingBox"]["Width"]> maxLeft:
            maxLeft = detect["Geometry"]["BoundingBox"]["Left"] + detect["Geometry"]["BoundingBox"]["Width"]

        if detect["Geometry"]["BoundingBox"]["Top"] < minTop:
            minTop = detect["Geometry"]["BoundingBox"]["Top"]

        if detect["Geometry"]["BoundingBox"]["Top"] + detect["Geometry"]["BoundingBox"]["Height"] > maxTop:
            maxTop = detect["Geometry"]["BoundingBox"]["Top"] + detect["Geometry"]["BoundingBox"]["Height"]


        width = (detect["Geometry"]["BoundingBox"]["Width"]) / np.float64(len(numbers))

        for index, number in enumerate(numbers):
            digit = (number,detect["Geometry"]["BoundingBox"]["Top"],detect["Geometry"]["BoundingBox"]["Left"] + (np.float64(index)*width))

            digits.append(digit)
#print(digits)
w, h = 9, 9
grid = [[0 for x in range(w)] for y in range(h)]
#print(grid)
width = (maxLeft - minLeft) / 9
height = (maxTop - minTop) / 9

for digit in digits:
    top = digit[1] + (height/4)
    left = digit[2] + (width/4)
    for i in range(9):
        if minTop + (np.float64(i)*height) < top and top < minTop + (np.float64(i+1)*height):
            for j in range(9):
                if minLeft + (np.float64(j) * width) < left and left < minLeft + (np.float64(j + 1) * width):

                    grid[i][j] = digit[0]
                    break
            break

rows = []
for i in range(9):
    row = ""
    for j in range(9):
        row = grid[i][j]
    rows.append(row)

partial_sudoku = np.array(grid)


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None

print_board(partial_sudoku)
solve(partial_sudoku)
print("___________________")
print()
print()
print()
print_board(partial_sudoku)

#https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
