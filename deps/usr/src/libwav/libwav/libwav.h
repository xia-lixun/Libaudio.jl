#pragma once
#include <stdint.h>


#if defined(LIBWAV_EXPORT) // inside DLL
	#define LIBWAV_API   __declspec(dllexport)
#else // outside DLL
	#define LIBWAV_API   __declspec(dllimport)
#endif  // XYZLIBRARY_EXPORT




extern "C" LIBWAV_API int64_t wavinfo(char * filepath, int64_t * meta);
extern "C" LIBWAV_API int64_t wavread(char * filepath, float * dat);
extern "C" LIBWAV_API int64_t wavwrite(char * filepath, float * dat, int64_t nf, int64_t ch, int64_t fs, int64_t bps, double t0, double t1);
extern "C" LIBWAV_API int64_t wavmeta(char * filepath);