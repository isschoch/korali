#include "auxiliar/fs.hpp"
#include "string.h"
#include <dirent.h>
#include <string>
#include <sys/stat.h>

namespace korali
{
void mkdir(const std::string dirPath)
{
  char tmp[256];
  char *p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp), "%s", dirPath.c_str());
  len = strlen(tmp);
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++)
    if (*p == '/')
    {
      *p = 0;
      ::mkdir(tmp, S_IRWXU);
      *p = '/';
    }
  ::mkdir(tmp, S_IRWXU);
}

std::vector<std::string> listDirFiles(const std::string dirPath)
{
  struct dirent *entry = nullptr;
  std::vector<std::string> dirList;

  DIR *dp = nullptr;

  dp = opendir(dirPath.c_str());
  if (dp != nullptr)
    while ((entry = readdir(dp)))
    {
      std::string filePath = dirPath;
      filePath += "/";
      filePath += entry->d_name;
      dirList.push_back(filePath);
    }
  closedir(dp);

  return dirList;
}

bool dirExists(const std::string dirPath)
{
  DIR *dir = opendir(dirPath.c_str());
  if (dir)
  {
    closedir(dir);
    return true;
  }

  return false;
}

} // namespace korali
