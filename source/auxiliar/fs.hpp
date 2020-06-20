/** \file
* @brief Contains auxiliar code for file system (files and folders) manipulation.
******************************************************************************/

#ifndef _AUXILIAR_FS_HPP_
#define _AUXILIAR_FS_HPP_

#include <string>
#include <vector>

namespace korali
{
/**
 * @brief Creates a new folder and builds the entire path, if necessary.
 * @param dirPath relative path to the new folder.
 */
void mkdir(const std::string dirPath);

/**
 * @brief Lists all files within within a given folder path
 * @param dirPath relative path to the folder to list.
 * @return A list with the path of all files found.
 */
std::vector<std::string> listDirFiles(const std::string dirPath);

/**
  * @brief Checks if directory exists
  * @param dirPath relative path to the folder to list.
  * @return true, if the directory exists, or; false, if it does not.
  */
bool dirExists(const std::string dirPath);

} // namespace korali

#endif // _AUXILIAR_FS_HPP_
