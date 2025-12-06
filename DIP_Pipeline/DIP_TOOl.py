import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tkinter as tk
from PIL import Image,ImageTk
from scipy.fft import fft2,ifft2
from scipy.signal import convolve2d
from tkinter import filedialog

orig_img=None
working_img=None
history=[]
tk_img=None


root=tk.Tk()
root.title("Image Processing")
root.geometry("1920x1080")
label=tk.Label(root,text="Welcome to DIP",font=("Arial", 20, "bold"))
label.place(relx=0.5, y=5, anchor="n")

canvas=tk.Canvas(root,width=600,height=600,bg='gray')
canvas.place(x=20,y=50)

path_entry=tk.Entry(root,width=50,font=("Arial",14))
path_entry.place(x=650,y=60)
placeholder="Enter Image path here"
path_entry.insert(0,placeholder)
path_entry.config(fg="gray")


label_status=tk.Label(root,text="",font=("Arial",15, "bold"))
label_status.place(x=160,y=680)
label_status2=tk.Label(root,text="",font=("Arial",15, "bold"))
label_status2.place(x=160,y=720)




def on_focus_out(event):
    if path_entry.get()=="":
        path_entry.insert(0,placeholder)
        path_entry.config(fg='gray')


def on_entry_click(event):
    if path_entry.get()==placeholder:
        path_entry.delete(0,"end")
        path_entry.config(fg="black")


def on_load_button_click():
    path=path_entry.get()
    load_image(path)


def load_image(path):
    global orig_img,working_img
    label_status2.config(text="",fg='blue')
    img=cv.imread(path)
    
    if img is None:
        print("Image was not found at path")
        label_status.config(text="Image was not found at path!",fg='red')
        label_status2.config(text="",fg='blue')
        return

    orig_img=img.copy()
    working_img=img.copy()
    label_status.config(text="Image Loadded Successfully!",fg='green')
    label_status2.config(text="Click on Current/Original button to show Image",fg='blue')
    

def show_on_canvas(img):
    canvas.delete("all")  # Add at the start of show_on_canvas() 
    global tk_img 
    if len(img.shape)==2: 
        img=cv.cvtColor(img,cv.COLOR_GRAY2RGB) 
    else: 
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB) 
    img=cv.resize(img,(600,600)) 
    pil_img=Image.fromarray(img) 
    tk_img=ImageTk.PhotoImage(pil_img) 
    canvas.create_image(0,0,anchor='nw',image=tk_img)
    label_status2.config(text="",fg='blue')


def crop_img():
    global history,working_img
    
    working_img=cv.resize(working_img,(600,600))
    if working_img is None:
        print("Load an Image first")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return
    history.append(working_img.copy())

    roi=cv.selectROI("Select image and  press enter or space",working_img)
    x,y,w,h=roi
    if w == 0 or h == 0:
        print("Cropping cancelled")
        cv.destroyWindow("Select ROI and press Enter or Space")
        return
    working_img=working_img[y:y+h,x:x+w]
    cv.destroyWindow("Select ROI and press Enter or Space")
    label_status.config(text="Image Cropped Successfully!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def undo_last():
    global history,working_img
    
    if history:
        working_img=history.pop()
        print("Change Reverted")
        label_status.config(text="Operation Reverted",fg='green')
        label_status2.config(text="Click on Current button to show Image",fg='blue')

    else:
        print("No changes to undo")
        label_status.config(text="No Changes to Revert",fg='red')
        label_status2.config(text="",fg='blue')


def show_info_window():
    info_win = tk.Toplevel(root)
    info_win.title("Software Guide & Warnings")
    info_win.geometry("600x500")

    text_area = tk.Text(info_win, wrap="word", font=("Arial", 11))
    text_area.pack(expand=True, fill="both")

    info_text = """
IMAGE PROCESSING TOOL - USER GUIDE
----------------------------------

1. LOADING & VIEWING IMAGES
   - Use 'Load Image' to import an image.
   - 'Original' button shows the original unmodified image.
   - 'Current' button shows the latest version after all applied filters.

2. APPLYING FILTERS
   - You may apply filters one after another.
   - Each filter will save the previous state in History.
   - Use 'Undo' to revert to the previous state.

3. IMPORTANT WARNINGS (AVOID ERRORS)
   • LOG TRANSFORM:
       - Only works on non-negative pixel values.
       - If the image contains negative values, you may get errors.
      
   • GAMMA TRANSFORM:
       - Gamma < 1 brightens image.
       - Gamma > 1 darkens image.
       - Avoid extremely high gamma values (>10).
       - If gamma value is invalid, empty, or negative, the default γ=0.3 is used.

   • K-MEANS SEGMENTATION:
       - K must be an integer.
       - K should ideally be between 2 and 10 for best results.
       - Works on original or current image.
       - If K is invalid, empty, or less than 1, the default K=3 is used.

   • RESTORATION FILTERS (CLSF / WIENER):
       - Perfect restoration is not possible if image has heavy noise.
       - Make sure image is grayscale before using some advanced filters.

4. RECOMMENDED WORKFLOW
   1. Load image.
   2. View the Original.
   3. Apply filters carefully.
   4. Use 'Undo' if results are not satisfactory.
   5. Use 'Save Image' only after checking final output.

5. NOTES
   - The software never overwrites your original image.
   - All operations are done on a working copy.
   - Undo stack stores all steps while using 'Current' mode.


"""


    text_area.insert("1.0", info_text)
    text_area.config(state="disabled")


def grayscale():
    global working_img, history
    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    history.append(working_img.copy())

    # Convert to grayscale if image is color
    if len(working_img.shape) == 3:
        working_img = cv.cvtColor(working_img, cv.COLOR_BGR2GRAY)

    label_status.config(text="Grayscale Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')

    
def negative_transformation():
    global working_img, history

    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    history.append(working_img.copy())

    # If grayscale
    if len(working_img.shape) == 2:
        working_img = 255 - working_img
        return

    # If color
    working_img = cv.bitwise_not(working_img)
    label_status.config(text="Negative Transformation Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def log_transform():
    global working_img, history

    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    history.append(working_img.copy())

    img = working_img.copy()

    # Convert color → grayscale only if needed
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log1p(img)

    working_img = np.uint8(log_img)
    label_status.config(text="Log Transform Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def gamma_transform():
    global working_img, history
    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="", fg='blue')
        return

    DEFAULT_GAMMA = 0.3

    # Parse user input
    try:
        gamma_val = float(gamma_entry.get().strip())
        if gamma_val <= 0:
            raise ValueError
        valid = True
    except:
        gamma_val = DEFAULT_GAMMA
        valid = False

    # Apply transform
    history.append(working_img.copy())

    img = working_img.astype(np.float32) / 255.0
    img = np.power(img, gamma_val)
    working_img = np.uint8(img * 255.0)

    # Show messages
    if valid:
        label_status.config(text=f"Gamma Transform Applied with γ={gamma_val}!", fg='green')
    else:
        label_status.config(text=f"Invalid gamma — using default γ={DEFAULT_GAMMA}", fg='orange')

    label_status2.config(text="Click on Current button to show Image", fg='blue')


    # Save for undo
    history.append(working_img.copy())

    # Apply gamma correction
    img = working_img.astype(np.float32) / 255.0
    img = np.power(img, gamma_val)
    img = img * 255.0

    working_img = np.uint8(np.clip(img, 0, 255))

    label_status.config(text=f"Gamma Transform Applied (γ={gamma_val})", fg='green')
    label_status2.config(text="Click on Current button to show Image", fg='blue')


def thresholding():
    global working_img, history

    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    history.append(working_img.copy())

    img = working_img

    # Convert color → grayscale before threshold
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, working_img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    label_status.config(text="Binary Thresholding Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')
    

def canny_edge_detection():
    global working_img,history
    if working_img is None:
        print("Load an image first")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return
    history.append(working_img.copy())
    img = working_img

    # Convert color → grayscale
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    working_img=cv.Canny(img,100,200)
    label_status.config(text="Canny Edge Detection Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def sobel_edge_detection():
    global working_img,history
    if working_img is None:
        print("Load an image first")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return
    history.append(working_img.copy())
    img = working_img

    # Convert color → grayscale
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobelx=cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely=cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    magnitude=cv.magnitude(sobelx,sobely)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    working_img=np.array(magnitude,dtype=np.uint8)
    label_status.config(text="Sobel Edge Detection Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def prewitt_edge_detection():
    global working_img,history
    if working_img is None:
        print("Load an image first")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return
    history.append(working_img.copy())

    img = working_img

    # Convert color → grayscale
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernelx=np.array([[-1,0,1],
             [-1,0,1],
             [-1,0,1]])
    kernely=np.array([[-1,-1,-1],
             [0,0,0],
             [1,1,1]])

    prewittx=cv.filter2D(img,-1,kernelx)
    prewitty=cv.filter2D(img,-1,kernely)
    magnitude=cv.magnitude(prewittx.astype(np.float32),prewitty.astype(np.float32))
    magnitude=cv.normalize(magnitude,None,0,255,cv.NORM_MINMAX)
    working_img=np.array(magnitude,dtype=np.uint8)
    label_status.config(text="Prewitt Edge Detection Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def histogram_equalization():
    global working_img, history

    if working_img is None:
        print("Load an image first")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    # Save previous state for undo
    history.append(working_img.copy())

    # Convert to grayscale if needed
    if len(working_img.shape) == 3 and working_img.shape[2] == 3:
        gray = cv.cvtColor(working_img, cv.COLOR_BGR2GRAY)
    else:
        gray = working_img.copy()

    # Histogram
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0,256])

    # PDF
    pdf = hist / hist.sum()

    # CDF
    cdf = np.cumsum(pdf)

    # Mapping
    mapping = np.round(cdf * 255).astype(np.uint8)

    # Apply mapping to each pixel
    equalized = mapping[gray]

    # Update working image (single channel)
    working_img = equalized
    label_status.config(text="Histogram Equalization applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def show_histogram_graphs():
    global orig_img, working_img

    if working_img is None:
        print("No image to show histogram for")
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    # Convert to grayscale if needed
    if len(orig_img.shape) == 3:
        orig_gray = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)
    else:
        orig_gray = orig_img.copy()

    equalized_gray = working_img.copy()

    # Histograms
    hist_orig, _ = np.histogram(orig_gray.flatten(), bins=256, range=[0,256])
    hist_eq, _ = np.histogram(equalized_gray.flatten(), bins=256, range=[0,256])

    label_status.config(text="Showing Graphs!", fg='green')

    # PLOTS
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0][0].imshow(orig_gray, cmap="gray")
    axes[0][0].set_title("Original Image")
    axes[0][0].axis("off")

    axes[0][1].imshow(equalized_gray, cmap="gray")
    axes[0][1].set_title("Equalized Image")
    axes[0][1].axis("off")

    axes[1][0].plot(hist_orig)
    axes[1][0].set_title("Original Histogram")

    axes[1][1].plot(hist_eq)
    axes[1][1].set_title("Equalized Histogram")

    plt.tight_layout()
    plt.show()


def clsf_restoration(gamma=1e-6):
    """
    Apply Constrained Least Squares Filter (CLSF) restoration
    on the current working_img.
    """
    global working_img, history

    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    # Save current state
    history.append(working_img.copy())

    # Convert to grayscale if needed
    img_arr = working_img.copy()
    if len(img_arr.shape) == 3:
        img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
    img_arr = img_arr.astype(np.float32)

    # PSF (point spread function)
    psf = np.ones((9, 9), dtype=np.float32) / 81

    # CLSF restoration
    G = fft2(img_arr)
    H = fft2(psf, s=img_arr.shape)

    # Laplacian operator for regularization
    P = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]], dtype=np.float32)
    P = fft2(P, s=img_arr.shape)

    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + gamma * np.abs(P)**2)) * G

    restored = np.abs(ifft2(F_hat))
    working_img = np.uint8(np.clip(restored, 0, 255))

    label_status.config(text="Constrained Least Squares Filter Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def wiener_restoration(K=0.001):
    """
    Apply Wiener filter restoration on the current working_img.
    """
    global working_img, history

    if working_img is None:
        label_status.config(text="Load an Image First", fg='red')
        label_status2.config(text="",fg='blue')
        return

    history.append(working_img.copy())

    img_arr = working_img.copy()
    if len(img_arr.shape) == 3:
        img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
    img_arr = img_arr.astype(np.float32)

    # PSF (point spread function)
    psf = np.ones((9, 9), dtype=np.float32) / 81
    H = fft2(psf, s=img_arr.shape)
    H_conj = np.conj(H)

    # FFT of degraded image
    G = fft2(img_arr)

    # Wiener filter formula
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G
    restored = np.abs(ifft2(F_hat))
    working_img = np.uint8(np.clip(restored, 0, 255))

    label_status.config(text="Wiener Filter Applied!", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def kmeans_segmentation(mode="current"):
    global orig_img, working_img, history

    # Select source image
    if mode == "original":
        if orig_img is None:
            print("Load an image first")
            return
        src_img = orig_img.copy()
    else:  # mode == "current"
        if working_img is None:
            print("Load an image first")
            return
        src_img = working_img.copy()
    DEFAULT_K = 3
    try:
        K = int(k_entry.get().strip())
        if K < 1:
            K = DEFAULT_K
    except:
        K = DEFAULT_K
    # Get K
   
    # Save history for undo
    history.append(working_img.copy())

    # Convert to RGB
    img_rgb = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)

    # Prepare data
    Z = img_rgb.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-means
    _, labels, centers = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_img = segmented.reshape(img_rgb.shape)

    # Store result in working_img
    working_img = cv.cvtColor(segmented_img, cv.COLOR_RGB2BGR)

    print(f"K-means applied on {mode.upper()} image (K={K}).")
    print("Click 'Show Current Image' to display.")
    label_status.config(text=f"K-means applied on {mode.upper()} image (K={K}).", fg='green')
    label_status2.config(text="Click on Current button to show Image",fg='blue')


def export_current_image():
    global working_img
    if working_img is None:
        label_status.config(text="No image to export!", fg='red')
        return
    
    # Ask user for file path
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Current Image As"
    )
    
    if file_path:
        # Save the working image
        cv.imwrite(file_path, working_img)
        label_status.config(text=f"Image saved to {file_path}", fg='green')

label_status2.config(text="")
path_entry.bind("<FocusIn>",on_entry_click)
path_entry.bind("<FocusOut>",on_focus_out)


load_button=tk.Button(root,text="Load Image",font=("Arial",10),command=on_load_button_click)
load_button.place(x=1250,y=60)


label_show=tk.Label(root,text="Show Image",font=("Arial",15, "bold"))
label_show.place(x=650,y=120)
orig_button=tk.Button(root,text="original Image",font=("Arial",10),command=lambda:show_on_canvas(orig_img))
orig_button.place(x=650,y=160)

curr_button=tk.Button(root,text="Current Image",font=("Arial",10),command=lambda:show_on_canvas(working_img))
curr_button.place(x=770,y=160)
gray_button=tk.Button(root,text="Gray Image",font=("Arial",10),command=grayscale)
gray_button.place(x=890,y=160)


label_util=tk.Label(root,text="Utility ",font=("Arial",15, "bold"))
label_util.place(x=650,y=220)


revert_button=tk.Button(root,text="Revert Current Operation",font=("Arial",10),command=undo_last)
revert_button.place(x=650,y=260)
revert_button=tk.Button(root,text="Crop Image",font=("Arial",10),command=crop_img)
revert_button.place(x=850,y=260)
info_button = tk.Button(root, text="Info / Help", font=("Arial", 10),command=lambda: show_info_window())
info_button.place(x=1050,y=260)

label_intensity_transform=tk.Label(root,text="Intensity Transformation",font=("Arial",15, "bold"))
label_intensity_transform.place(x=650,y=320)

Thresh_button=tk.Button(root,text="Binary Thresholding",font=("Arial",10),command=thresholding)
Thresh_button.place(x=650,y=360)

neg_button=tk.Button(root,text="Negative Transformation",font=("Arial",10),command=negative_transformation)
neg_button.place(x=850,y=360)

log_button=tk.Button(root,text="Log Transformation",font=("Arial",10),command=log_transform)
log_button.place(x=1050,y=360)


label_gamma = tk.Label(root, text="Gamma:", font=("Arial", 12, "bold"))
label_gamma.place(x=850, y=400)

gamma_entry = tk.Entry(root, width=5, font=("Arial", 14))
gamma_entry.place(x=950, y=400)

gam_button=tk.Button(root,text="Gamma Transformation",font=("Arial",10),command=gamma_transform)
gam_button.place(x=650,y=400)



label_edge=tk.Label(root,text="Edge Detection",font=("Arial",15, "bold"))
label_edge.place(x=650,y=460)

canny_button=tk.Button(root,text="Canny Edge Detection",font=("Arial",10),command=canny_edge_detection)
canny_button.place(x=650,y=500)
sobel_button=tk.Button(root,text="Sobel Edge Detection",font=("Arial",10),command=sobel_edge_detection)
sobel_button.place(x=850,y=500)
prewitt_button=tk.Button(root,text="Prewitt Edge Detection",font=("Arial",10),command=prewitt_edge_detection)
prewitt_button.place(x=1050,y=500)


label_hist=tk.Label(root,text="Histogram",font=("Arial",15, "bold"))
label_hist.place(x=650,y=560)

equalized_button=tk.Button(root,text="Hisogram Equalization",font=("Arial",10),command=histogram_equalization)
equalized_button.place(x=650,y=600)
ohist_button=tk.Button(root,text="Histogram Graphs",font=("Arial",10),command=show_histogram_graphs)
ohist_button.place(x=850,y=600)

label_filter=tk.Label(root,text="Filters",font=("Arial",15, "bold"))
label_filter.place(x=1050,y=560)

clsf_button=tk.Button(root,text="Constrained Least Square Filter",font=("Arial",10),command=clsf_restoration)
clsf_button.place(x=1050,y=600)
wiener_button=tk.Button(root,text="Wiener Filter",font=("Arial",10),command=wiener_restoration)
wiener_button.place(x=1300,y=600)


label_cluster=tk.Label(root,text="Image Clustering",font=("Arial",15, "bold"))
label_cluster.place(x=650,y=640)
currk_button=tk.Button(root,text="Current",font=("Arial",10), command=lambda: kmeans_segmentation(mode="current"))
currk_button.place(x=650,y=680)
colork_button=tk.Button(root,text="Color(Original)",font=("Arial",10), command=lambda: kmeans_segmentation(mode="original"))
colork_button.place(x=770,y=680)

label_cno=tk.Label(root,text="K: ",font=("Arial",12, "bold"))
label_cno.place(x=900,y=680)
k_entry=tk.Entry(root,width=5,font=("Arial",14))
k_entry.place(x=950,y=680)


export_button = tk.Button(root, text="Export Current Image", font=("Arial", 10), command=export_current_image)
export_button.place(x=650, y=750)
root.mainloop()

