import subprocess
import sys
import os

def main():
    if (len(sys.argv) != 2):
        print('Usage: python feture_detection.py <filename>')
        return -1
    fullname = sys.argv[1]
    filename, extension = os.path.splitext(fullname)
    filename = os.path.basename(filename)
    if (extension != '.pcd'):
        print('File type must be .pcd')
        return -1
    #p1 = subprocess.Popen([r"C:\Users\karol\OneDrive\Lõputöö\pcd2txt\x64\Debug\pcd2txt.exe", fullname])
    #p1.wait()
    print("PCD file read")
    cwd = os.getcwd()
    txt_path = os.path.join(cwd, filename + ".txt")
    subprocess.call(["python", "./python/xyz2png.py", txt_path])
    print("Image generated")
    png_path = os.path.join(cwd, filename + ".png")
    subprocess.call(["python", "./python/pcd_feature_extraction.py", png_path])
    print("Detection completed")

if __name__ == "__main__":
    main()
