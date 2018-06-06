import subprocess
import sys
import os

def main():
	if (len(sys.argv) != 2):
		print('Usage: python feature_extraction.py <filename>')
		return -1
	fullname = sys.argv[1]
	filename, extension = os.path.splitext(fullname)
	filename = os.path.basename(filename)
	if (extension != '.pcd') and (extension != '.bmp'):
		print('File type must be .pcd or .bmp')
		return -1
	cwd = os.getcwd()
	if extension == '.pcd':
		p1 = subprocess.Popen([r"C:\Users\karol\OneDrive\Lõputöö\pcd2txt\x64\Debug\pcd2txt.exe", fullname])
		p1.wait()
		print("PCD file read")
		txt_path = os.path.join(cwd, filename + ".txt")
		subprocess.call(["python3", "./python/txt_to_bmp_converter.py", txt_path])
		print("Image generated")
	im_path = os.path.join(cwd, filename + ".bmp")
	subprocess.call(["python3", "./python/pcd_feature_extraction.py", im_path])
	print("Detection completed")

if __name__ == "__main__":
	main()