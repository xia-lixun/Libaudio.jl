#pragma once

#if defined(PA_DYNAMIC_EXPORT) // inside DLL
	#define PA_DYNAMIC_API   __declspec(dllexport)
#else // outside DLL
	#define PA_DYNAMIC_API   __declspec(dllimport)
#endif  // XYZLIBRARY_EXPORT




extern "C" PA_DYNAMIC_API int play(const char * path, int sample_rate);
extern "C" PA_DYNAMIC_API int record(const char * path, int sample_rate, int channels, double duration, int bits);
extern "C" PA_DYNAMIC_API int playrecord(const char * path_play, const char * path_record, int sample_rate, int channels_record, int bits_record);
