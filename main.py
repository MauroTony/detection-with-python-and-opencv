import cv2

if __name__ == "__main__":
    img = cv2.imread('Imagens\\pessoas.jpg')
    cascade_smile = cv2.CascadeClassifier('Cascades\\haarcascade_smile.xml')
    cascade_face = cv2.CascadeClassifier(
        'Cascades\\haarcascade_frontalface_default.xml')
    cascade_eye = cv2.CascadeClassifier('Cascades\\haarcascade_eye.xml')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(
        img_gray, scaleFactor=1.25, minSize=(100, 100))
    smiles = cascade_smile.detectMultiScale(
        img_gray, scaleFactor=1.2, minSize=(50, 20))
    eyes = cascade_eye.detectMultiScale(img_gray)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for x_s, y_s, w_s, h_s in smiles:
            if x <= x_s and y <= y_s and x + w >= x_s + w_s and y + h >= y_s + h_s:
                cv2.rectangle(img, (x_s, y_s), (x_s + w_s,
                                                y_s + h_s), (0, 255, 0), 1)
        for x_s, y_s, w_s, h_s in eyes:
            if x <= x_s and y <= y_s and x + w >= x_s + w_s and y + h >= y_s + h_s:
                cv2.rectangle(img, (x_s, y_s), (x_s + w_s,
                                                y_s + h_s), (0, 0, 255), 1)

    cv2.imshow("", img)
    cv2.waitKey()
