#pragma once
#include <cstdbool>
#include <cinttypes>
#include <cstdio> 



typedef struct MetaInfoStruct 
{
	//RIFF master header
	uint8_t riff_ckID[4];
	uint32_t riff_cksize;
	uint8_t riff_waveID[4];

	//FORMAT header
	uint8_t fmt_ckID[4];
	uint32_t fmt_cksize;
	uint16_t fmt_wFormatTag;
	uint16_t fmt_nChannels;
	uint32_t fmt_nSamplesPerSec;
	uint32_t fmt_nAvgbytesPerSec;
	uint16_t fmt_nBlockAlign;
	uint16_t fmt_wBitsPerSample;

	uint16_t fmt_cbSize;
	//EXTENDED FORMAT(ONLY FOR READ)
	uint16_t fmt_wValidBitsPerSample;
	uint32_t fmt_dwChannelMask;
	uint8_t fmt_SubFormat[16];

	//FACT header
	uint8_t fact_ckID[4];
	uint32_t fact_cksize;
	uint32_t fact_dwSampleLength; //number of samples per channel

	uint32_t unknown_cksize;
								  
	uint8_t data_ckID[4];//DATA
	uint32_t data_cksize;

} MetaInfo_t;






//application scenarios: 
// (a) Cwav a; a.GetFrameMatrix("file.wav"); f(FrameMatrix); Save2File_XXX();
// (b) Cwav b; b.SetFrameMatrix(1*fs, 2, 44100); f(FrameMatrix); Save2File_XXX();
class CsWav
{

public:
	CsWav();
	virtual ~CsWav();

	size_t ExtractMetaInfo(const char * FilePath);
	void PrintMetaInfo(void);
	
	size_t GetFrameLength(void) const;
	size_t GetNumChannel(void) const;
	size_t GetSampleRate(void) const;
	size_t GetBitsPerSample(void) const;
	
	float * GetFrameMatrix(const char * FilePath);                     //compose a framematrix from file
	float * SetFrameMatrix(size_t nFrame, size_t nChannel, size_t nSampleRate); //compose a framematrix from param

	size_t SaveFile(const char * FilePath, double Start, double Stop, size_t BitsPerSample);
	size_t SaveFile(const char * FilePath, size_t BitsPerSample);


	//2016-11-30: add file stream service for extra-large file processing
	float * OpenStreamFrom(const char * FilePath);
	float * OpenStreamTo(const char * FilePath, size_t nFrame, size_t nChannel, size_t nSampleRate, size_t BitsPerSample);
	void ReadFrameFromStream();
	void WriteFrameToStream();
	void CloseStream();
	float * GetFrameVector();




private:
	size_t ExtractData_16b(const char * FilePath);
	size_t ExtractData_24b(const char * FilePath);
	size_t ExtractData_flt(const char * FilePath);

	void MakeMetaInfo(size_t nChannel, size_t nSampleRate, size_t nSample, size_t fmt);
	size_t SaveMetaInfo(const char * FilePath);

	//write framematrix to files with slice param
	size_t Save2File_flt(const char * FilePath, double Start, double Stop);
	size_t Save2File_16b(const char * FilePath, double Start, double Stop);
	size_t Save2File_24b(const char * FilePath, double Start, double Stop);

	//write framematrix to files entirely
	size_t Save2File_flt(const char * FilePath);
	size_t Save2File_16b(const char * FilePath);
	size_t Save2File_24b(const char * FilePath);

	//2016-11-30
	void OpenFileDescriptor(const char * FilePath);

	void ExtractFrame_16b();
	void ExtractFrame_24b();
	void ExtractFrame_flt();

	void SaveFrame_16b();
	void SaveFrame_24b();
	void SaveFrame_flt();



protected:
	MetaInfo_t MetaInfo;
	size_t NumChannel;
	size_t SampleRate;
	size_t NumFrame;
	float *FrameMatrix;

	//2016-11-30
	float *FrameVector;
	size_t StreamFrameCounter;
	FILE *StreamFID;
	int16_t SampleInt16[65536];
	size_t fmt;

};

