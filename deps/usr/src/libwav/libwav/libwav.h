#pragma once
#include <stdint.h>


#if defined(LIBWAV_EXPORT) // inside DLL
	#define LIBWAV_API   __declspec(dllexport)
#else // outside DLL
	#define LIBWAV_API   __declspec(dllimport)
#endif  // XYZLIBRARY_EXPORT




extern "C" LIBWAV_API int wavinfo(char * filepath, int64_t * meta);
extern "C" LIBWAV_API int wavread(char * filepath, float * dat);
