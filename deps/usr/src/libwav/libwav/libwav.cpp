#include <cstring>
#include <stdint.h>
#include "libwav.h"
#include "CsWav.h"




// c interface for dll 2018-09-12 (baby endures diarrhea :-\)
int wavinfo(char * filepath, int64_t * meta)
{
	CsWav w;
	w.ExtractMetaInfo(filepath);
	meta[0] = (int64_t)w.GetFrameLength();
	meta[1] = (int64_t)w.GetNumChannel();
	meta[2] = (int64_t)w.GetSampleRate();
	meta[3] = (int64_t)w.GetBitsPerSample();
	return 0;
}


int wavread(char * filepath, float * dat)
{
	CsWav w;
	w.ExtractMetaInfo(filepath);
	float * p = w.GetFrameMatrix(filepath); 
	memcpy(dat, p, sizeof(float) * w.GetFrameLength() * w.GetNumChannel());
	return 0;
}