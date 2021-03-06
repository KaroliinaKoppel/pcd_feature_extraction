#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>

using std::string;

string getFileName(const string& s) {

	char sep = '/';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		return(s.substr(i + 1, s.length() - i));
	}

	return("");
}

int
main(int argc, char** argv)
{
	// Check if argument is provided
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <filename>\n";
		return (-1);
	}

	// Extract file extention
	char* fullname = argv[1];
	std::string s = fullname;
	std::string extension = s.substr(s.find_last_of("."), std::string::npos);
	std::string filename = s.substr(0, s.find_last_of("."));

	// Check if correct file type is provided
	if (extension != ".pcd")
	{
		std::cout << "File type must be .pcd\n";
		return (-1);
	}

	// Initialize PointCloud
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// Try to read file
	std::cout << "Reading file\n";
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(fullname, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read the file\n");
		return (-1);
	}

	std::cout << "Loaded " 
		<< cloud->width * cloud->height
		<< " data points from " << fullname << std::endl;

	// Write file
	string output_name = getFileName(filename);
	std::ofstream out;
	std::string path = output_name + ".txt";
	out.open(path);

	std::cout << "Writing to file\n";
	for (size_t i = 0; i < cloud->points.size(); i = 10 + i)
	{
		if ((int)cloud->points[i].z > 100)
		{
			out << (int)cloud->points[i].x / 100 << " "
				<< (int)cloud->points[i].y / 100 * -1 << " "
				<< (int)cloud->points[i].z / 100 << "\n";
		}
	}
	out.close();

	return (0);
}

