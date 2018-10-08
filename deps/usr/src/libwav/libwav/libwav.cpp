#include <cstring>
#include <stdint.h>
#include "libwav.h"
#include "CsWav.h"




// c interface for dll 2018-09-12 (baby endures diarrhea :-\)
int64_t wavinfo(char * filepath, int64_t * meta)
{
	CsWav w;
	w.ExtractMetaInfo(filepath);
	meta[0] = (int64_t)w.GetFrameLength();
	meta[1] = (int64_t)w.GetNumChannel();
	meta[2] = (int64_t)w.GetSampleRate();
	meta[3] = (int64_t)w.GetBitsPerSample();
	return (int64_t)0;
}


int64_t wavread(char * filepath, float * dat)
{
	CsWav w;
	w.ExtractMetaInfo(filepath);
	float * p = w.GetFrameMatrix(filepath); 
	memcpy(dat, p, sizeof(float) * w.GetFrameLength() * w.GetNumChannel());
	return (int64_t)0;
}


int64_t wavwrite(char * filepath, float * dat, int64_t nf, int64_t ch, int64_t fs, int64_t bps, double t0, double t1)
{
	CsWav w;
	float * f = w.SetFrameMatrix((size_t)nf, (size_t)ch, (size_t)fs); //compose a framematrix from param
	memcpy(f, dat, sizeof(float) * w.GetFrameLength() * w.GetNumChannel());

	size_t b = 0;
	if(t0 == t1)
		b = w.SaveFile(filepath, (size_t)bps);
	else
		b = w.SaveFile(filepath, t0, t1, (size_t)bps);
	return (int64_t)b;
}		


int64_t wavmeta(char * filepath)
{
	CsWav w;
	w.ExtractMetaInfo(filepath);
	w.PrintMetaInfo();
	return (int64_t)0;
}