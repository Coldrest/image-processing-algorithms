from tkinter import filedialog
import cv2
import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tkinter import messagebox
from CTkMessagebox import CTkMessagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def show_c():
    CTkMessagebox(title="Adaptive Threshold",message="Her pikselin eşik değerini ayarlamak için kullanılır!", icon="question", option_1="Tamam")
def show_kalinlik():
    CTkMessagebox(title="Kenarlık Ekleme",message="Kenar genişliği belirlemek için kullanılır!", icon="question", option_1="Tamam")
def show_gamma():
    CTkMessagebox(title="Gamma Correction",message="Gama değerini ayarlamak için kullanılır (1.0 = orijinal, <1.0 = aydınlatma, >1.0 = karartma)!", icon="question", option_1="Tamam")
def show_kernel():
    CTkMessagebox(title="Blur, Median Blur, Box Filter, Gaussian Blur",message="Bulanıklaştırma işlemleri için kernel boyutunu belirlemede kullanılır!", icon="question", option_1="Tamam")

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Enes Arayüz")

        self.video_source = 0  # 0, varsayılan kamerayı temsil eder (gerektiğinde değiştirilebilir)
        self.cap = cv2.VideoCapture(self.video_source)

        # Başlangıçta kamera çözünürlüğünü al
        _, frame = self.cap.read()
        self.initial_width, self.initial_height = frame.shape[1], frame.shape[0]

        # Kamera çözünürlüğü ile canvas'ı başlat
        self.canvas = ctk.CTkCanvas(root, width=self.initial_width, height=self.initial_height, highlightthickness=0, bd=0)
        self.canvas.pack(fill=ctk.BOTH, expand=True)

        self.algorithms = ["Orijinal", "Adaptive Threshold", "Otsu Threshold", "Kenarlık Ekle", "Blur", "Median Blur", "Box Filter", "Bilateral Filter", "Gaussian Blur", "Görüntü Keskinleştirme", "Gamma Correction", "Histogram", "Histogram Eşitleme","Sobel Kenar Algoritması","Laplacian Kenar Algoritması","Canny Kenar Algoritması","Deriche Algoritması","Harris Köşe Algoritması","Yüz Tanıma","Contour Tespit Algoritması","Watershed Algoritması"]
        self.algorithm_var = ctk.StringVar()
        self.algorithm_var.set(self.algorithms[0])

        algorithm_dropdown = ctk.CTkComboBox(root, variable=self.algorithm_var, values=self.algorithms, state="readonly",width=180)
        algorithm_dropdown.pack(pady=10)

        self.param_c_label = ctk.CTkLabel(root, text="C Değeri(Girilmez ise 2):")
        self.param_c_label.pack()
        self.param_c_entry = ctk.CTkEntry(root, state=ctk.DISABLED)  # Başlangıçta devre dışı bırakılmış
        self.param_c_entry.pack()
        question_mark_icon_c = ctk.CTkButton(root, text="?", cursor="question_arrow",width=15,height=15,command=show_c)
        question_mark_icon_c.place(x=self.param_c_label.winfo_x()+320 + self.param_c_label.winfo_reqwidth() + 5, y=self.param_c_label.winfo_y()+437)

        self.param_kalinlik_label = ctk.CTkLabel(root, text="Kalınlık Değeri(Girilmez ise 20):")
        self.param_kalinlik_label.pack()
        self.param_kalinlik_entry = ctk.CTkEntry(root, state=ctk.DISABLED)  # Başlangıçta devre dışı bırakılmış
        self.param_kalinlik_entry.pack()
        question_mark_icon_kalinlik = ctk.CTkButton(root, text="?", cursor="question_arrow",width=15,height=15,command=show_kalinlik)
        question_mark_icon_kalinlik.place(x=self.param_c_label.winfo_x()+340 + self.param_c_label.winfo_reqwidth() + 5, y=self.param_c_label.winfo_y()+493)


        self.param_gamma_label = ctk.CTkLabel(root, text="Gamma Değeri(Girilmez ise 2):")
        self.param_gamma_label.pack()
        self.param_gamma_entry = ctk.CTkEntry(root, state=ctk.DISABLED)  # Başlangıçta devre dışı bırakılmış
        self.param_gamma_entry.pack()
        question_mark_icon_gamma = ctk.CTkButton(root, text="?", cursor="question_arrow",width=15,height=15,command=show_gamma)
        question_mark_icon_gamma.place(x=self.param_c_label.winfo_x()+339 + self.param_c_label.winfo_reqwidth() + 5, y=self.param_c_label.winfo_y()+549)

        self.param_kernel_label = ctk.CTkLabel(root, text="Kernel Değeri(Girilmez ise 5):")
        self.param_kernel_label.pack()
        self.param_kernel_entry = ctk.CTkEntry(root, state=ctk.DISABLED)  # Başlangıçta devre dışı bırakılmış
        self.param_kernel_entry.pack()
        question_mark_icon_kernel = ctk.CTkButton(root, text="?", cursor="question_arrow",width=15,height=15,command=show_kernel)
        question_mark_icon_kernel.place(x=self.param_c_label.winfo_x()+333 + self.param_c_label.winfo_reqwidth() + 5, y=self.param_c_label.winfo_y()+605.5)

        save_button = ctk.CTkButton(root, text="Kaydet", command=self.save_image)
        save_button.pack(pady=10)

        self.selected_algorithm = None  # Başlangıçta seçili algoritma olmasın
        self.after_id = None
        self.update()
    

    def process_image(self):
        if self.algorithm_var.get() != self.selected_algorithm:
            self.selected_algorithm = self.algorithm_var.get()

            # Seçilen algoritmayı baz alarak parametre girişlerini etkinleştir veya devre dışı bırak
            if self.selected_algorithm == "Adaptive Threshold":
                self.param_c_entry.configure(state=ctk.NORMAL)
                self.param_gamma_entry.configure(state=ctk.DISABLED)
                self.param_kalinlik_entry.configure(state=ctk.DISABLED)
                self.param_kernel_entry.configure(state=ctk.DISABLED)
            elif self.selected_algorithm == "Gamma Correction":
                self.param_c_entry.configure(state=ctk.DISABLED)
                self.param_gamma_entry.configure(state=ctk.NORMAL)
                self.param_kalinlik_entry.configure(state=ctk.DISABLED)
                self.param_kernel_entry.configure(state=ctk.DISABLED)
            elif self.selected_algorithm == "Kenarlık Ekle":
                self.param_c_entry.configure(state=ctk.DISABLED)
                self.param_gamma_entry.configure(state=ctk.DISABLED)
                self.param_kalinlik_entry.configure(state=ctk.NORMAL)
                self.param_kernel_entry.configure(state=ctk.DISABLED)
            elif self.selected_algorithm in ["Blur", "Median Blur", "Box Filter", "Gaussian Blur",]:
                self.param_c_entry.configure(state=ctk.DISABLED)
                self.param_gamma_entry.configure(state=ctk.DISABLED)
                self.param_kalinlik_entry.configure(state=ctk.DISABLED)
                self.param_kernel_entry.configure(state=ctk.NORMAL)
            else:
                self.param_c_entry.configure(state=ctk.DISABLED)
                self.param_gamma_entry.configure(state=ctk.DISABLED)
                self.param_kalinlik_entry.configure(state=ctk.DISABLED)
                self.param_kernel_entry.configure(state=ctk.DISABLED)


        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # Kameraya ayna efekti ver

        if self.selected_algorithm == "Adaptive Threshold":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            block_size = 11
            try:
                C = float(self.param_c_entry.get())
            except ValueError:
                # Girişin geçerli bir ondalık sayı olmadığı durumu ele al
                C = 2.0  # Default değer
            frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        elif self.selected_algorithm == "Otsu Threshold":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.selected_algorithm == "Kenarlık Ekle":
            border_color = (0, 0, 0)
            try:
                border_width = int(self.param_kalinlik_entry.get())
            except ValueError:
                # Girişin geçerli bir ondalık sayı olmadığı durumu ele al
                border_width = 20  # Default değer

            # Kameranın başlangıç çözünürlüğünü al
            height, width, _ = self.cap.read()[1].shape

            # Kenarlığın boyutu aşmamasını sağla
            border_width = min(border_width, height // 2, width // 2)

            # numpy kullanarak bir kenar oluştur
            top_border = np.full((border_width, width, 3), border_color, dtype=np.uint8)
            bottom_border = np.full((border_width, width, 3), border_color, dtype=np.uint8)
            left_border = np.full((height, border_width, 3), border_color, dtype=np.uint8)
            right_border = np.full((height, border_width, 3), border_color, dtype=np.uint8)

            # Kenarları çerçevenin içine sığdır
            frame[:border_width, :, :] = top_border
            frame[-border_width:, :, :] = bottom_border
            frame[:, :border_width, :] = left_border
            frame[:, -border_width:, :] = right_border

        elif self.selected_algorithm == "Blur":
            try:
                kernel_size = int(self.param_kernel_entry.get())
            except ValueError:
                # Girişin geçerli bir sayı olmadığı durumu ele al
                kernel_size = 5  # Default değer
            frame = cv2.blur(frame, (kernel_size, kernel_size))
        elif self.selected_algorithm == "Median Blur":
            kernel_size = self.get_kernel_size()
            kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1 
            frame = cv2.medianBlur(frame, kernel_size)

        elif self.selected_algorithm == "Box Filter":
            try:
                kernel_size = int(self.param_kernel_entry.get())
            except ValueError:
                # Girişin geçerli bir sayı olmadığı durumu ele al
                kernel_size = 5  # Default değer
            
            frame = cv2.boxFilter(frame, -1, (kernel_size, kernel_size))
        elif self.selected_algorithm == "Bilateral Filter":
            frame = cv2.bilateralFilter(frame, 9, 75, 75)   
        elif self.selected_algorithm == "Gaussian Blur":
            kernel_size = self.get_kernel_size()
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        elif self.selected_algorithm == "Görüntü Keskinleştirme":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        elif self.selected_algorithm == "Gamma Correction":
            try:
                gamma_value = float(self.param_gamma_entry.get())
            except ValueError:
                gamma_value = 2  # Default değer
            frame = self.apply_gamma_correction(frame, gamma=gamma_value)
        elif self.selected_algorithm == "Histogram":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title('Histogram Curve')
            plt.xlabel('Pixel Value')
            plt.ylabel('Pixel Count')
            plt.show()
             # Histogram penceresi kapatıldığında orijinal görüntüyü göster
            self.algorithm_var.set("Orijinal")
            self.selected_algorithm = "Orijinal"
            return
        elif self.selected_algorithm == "Histogram Eşitleme":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(frame)
            fig = plt.figure(facecolor='xkcd:dark grey')
            subplot1 = plt.subplot(1, 2, 1)
            subplot1.imshow(frame, cmap='gray')
            subplot1.set_title('Orijinal Görüntü', color='white')  # Set title color
            subplot2 = plt.subplot(1, 2, 2)
            subplot2.imshow(equalized_image, cmap='gray')
            subplot2.set_title('Histogram Eşitleme Sonrası', color='white')  # Set title color
            # Set axis label color
            for ax in fig.axes:
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            # Set tick color
            for tick in subplot1.get_xticklabels() + subplot1.get_yticklabels() + subplot2.get_xticklabels() + subplot2.get_yticklabels():
                tick.set_color('white')
            # Set text color for numbers
            for text in subplot1.texts + subplot2.texts:
                text.set_color('white')
            
            plt.show()
            self.algorithm_var.set("Orijinal")
            self.selected_algorithm = "Orijinal"
            return
        elif self.selected_algorithm == "Sobel Kenar Algoritması":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobelx = np.abs(sobelx)
            sobely = np.abs(sobely)
            edges = cv2.bitwise_or(sobelx, sobely)

            edges = cv2.convertScaleAbs(edges)

            self.display_image(edges)
            return
        elif self.selected_algorithm == "Laplacian Kenar Algoritması":
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Laplacian edge detection directly to the grayscale frame
            laplacian_result1 = cv2.Laplacian(gray_frame, cv2.CV_64F)
            laplacian_result1 = np.uint8(np.absolute(laplacian_result1))

            # Apply Laplacian edge detection after applying Gaussian blur to the frame
            img_blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0)
            laplacian_result2 = cv2.Laplacian(img_blurred, ddepth=-1, ksize=3)

            # Display the original frame, and the results of Laplacian edge detection
            f, eksen = plt.subplots(1, 3, figsize=(15, 5))
            eksen[0].imshow(gray_frame, cmap="gray")
            eksen[1].imshow(laplacian_result1, cmap="gray")
            eksen[2].imshow(laplacian_result2, cmap="gray")

            plt.show()

            self.algorithm_var.set("Orijinal")
            self.selected_algorithm = "Orijinal"

        elif self.selected_algorithm == "Canny Kenar Algoritması":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(frame_gray, low_threshold, high_threshold, L2gradient=True)

            self.display_image(edges)
            return
        elif self.selected_algorithm == "Deriche Algoritması":
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = self.deriche_edge_detection(frame_gray)

            self.display_image(edges)
            return
        elif self.selected_algorithm == "Harris Köşe Algoritması":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corner_quality = 0.04
            min_distance = 10
            block_size = 3

            corners = cv2.cornerHarris(frame_gray, block_size, 3, corner_quality)

            corners = cv2.dilate(corners, None)
            frame[corners > 0.01 * corners.max()] = [0, 0, 255]

            self.display_image(frame)
            return
        elif self.selected_algorithm == "Yüz Tanıma":
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=6)

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

            # Display the frame with face detection
            self.display_image(frame)
        elif self.selected_algorithm == "Contour Tespit Algoritması":
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            frame = self.contour_detection(frame)
            self.display_image(frame)
        elif self.selected_algorithm == "Watershed Algoritması":
            # Your new algorithm code
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            imgBlr = cv2.medianBlur(frame, 31)
            imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)
            _, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
            sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
            _, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sureFG = np.uint8(sureFG)
            unknown = cv2.subtract(sureBG, sureFG)
            markers = cv2.connectedComponents(sureFG)[1]
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(frame, markers)

            contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            imgCopy = frame.copy()
            for i in range(len(contours)):
                if hierarchy[0][i][3] == -1:
                    cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

            # Display the original frame and segmented images with a smaller window
            f, eksen = plt.subplots(3, 3, figsize=(15, 8))  # Adjust the figsize as needed
            eksen[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            eksen[0, 1].imshow(imgBlr, cmap='gray')
            eksen[0, 2].imshow(imgGray, cmap='gray')
            eksen[1, 0].imshow(imgTH, cmap='gray')
            eksen[1, 1].imshow(imgOPN, cmap='gray')
            eksen[1, 2].imshow(sureBG, cmap='gray')
            eksen[2, 0].imshow(dist_transform, cmap='viridis')
            eksen[2, 1].imshow(sureFG, cmap='gray')
            eksen[2, 2].imshow(cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB))

            # Adjust layout for better visualization
            plt.show()

            self.algorithm_var.set("Orijinal")
            self.selected_algorithm = "Orijinal"

            return


        if self.selected_algorithm != "Histogram":
            self.display_image(frame)
        if self.selected_algorithm != "Histogram Eşitleme":
            self.display_image(frame)

    def apply_gamma_correction(self, image, gamma=1.0):
        image_normalized = image / 255.0
        gamma_corrected = np.power(image_normalized, gamma)
        gamma_corrected = np.uint8(gamma_corrected * 255)
        return gamma_corrected
    def deriche_edge_detection(self, frame):
        alpha = 0.5
        kernel_size = 3

        kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
        deriche_kernel_x = alpha * kx
        deriche_kernel_y = alpha * ky

        deriche_x = cv2.filter2D(frame, -1, deriche_kernel_x)
        deriche_y = cv2.filter2D(frame, -1, deriche_kernel_y)

        edges = np.sqrt(deriche_x**2 + deriche_y**2)

        # Manually scale values to the range [0, 255]
        edges = (edges / np.max(edges) * 255).astype(np.uint8)

        return edges

    def get_kernel_size(self):
        try:
            kernel_size = int(self.param_kernel_entry.get())
            # Kernel hep tek sayı
            kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        except ValueError:
            kernel_size = 5  # Default değeri

        return int(kernel_size)
    
    def contour_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        return frame
    
    
    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(frame)

        # Önceki içeriği canvas üzerinden temizle
        self.canvas.delete("all")

        # Orta kısmı hesapla
        x = (self.initial_width - frame.width) // 2
        y = (self.initial_height - frame.height) // 2

        self.canvas.create_image(x, y, anchor=ctk.NW, image=photo)
        self.canvas.photo = photo

    def save_image(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        self.process_image()

        frame_rgb = self.get_processed_image()

        pil_image = Image.fromarray(frame_rgb)

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])

        if file_path:
            pil_image.save(file_path)
            print(f"Processed Image saved to {file_path}")

    def get_processed_image(self):
        image = self.canvas.photo

        pil_image = ImageTk.getimage(image)

        frame_rgb = np.array(pil_image)

        return frame_rgb

    def update(self):
        self.process_image()
        # Boyut değiştirme olmadan görüntüyü periyodik olarak güncelle
        self.after_id = self.root.after(33, self.update)  # Her 33 milisaniyede bir güncelle (30 fps)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.after_id:
            self.root.after_cancel(self.after_id)
    

if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageProcessor(root)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Get the screen width
    screen_width = root.winfo_screenwidth()

    # Calculate the x coordinate to center the window along the x-axis
    x = (screen_width - root.winfo_reqwidth()) // 2

    # Set the window position
    root.geometry("+{}+{}".format(x, 0))
    root.resizable(width=False, height=False)
    root.mainloop()
