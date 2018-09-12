/** @file PaTask.cpp
	@ingroup PaDynamic
	@brief Portaudio wrapper for dynamic languages: Julia/Matlab etc.
	@author Lixun Xia <lixun.xia2@harman.com>
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cassert>
#include "./CsWav.h"
#include "../include/portaudio.h"
#include "../include/pa_asio.h"
#include "./PaTask.h"


#define FRAMES_PER_BUFFER	(64)




class PaPlayRecord
{
public:
	PaPlayRecord(
		std::string path_to_play,
		std::string path_to_record,
		size_t process_sample_rate,
		size_t channels,
		size_t bits_per_sample
	) : stream(0), mic_cntframes(0), mic_dat(NULL), spk_frame(0)
	{
		//init playback wav source
		size_t head_bytes = spk_wav.ExtractMetaInfo(path_to_play.c_str());
		spk_wav.PrintMetaInfo();
		assert(spk_wav.GetSampleRate() == process_sample_rate);
		sample_rate = process_sample_rate;
		spk_totsps = spk_wav.GetFrameLength() * spk_wav.GetNumChannel();
		spk_dat = spk_wav.GetFrameMatrix(path_to_play.c_str());

		//init recording wav source
		bits_persample = bits_per_sample;
		path_written = path_to_record;
		mic_channels = channels;
		mic_totframes = spk_wav.GetFrameLength();
		mic_dat = mic_wav.SetFrameMatrix(mic_totframes, mic_channels, sample_rate);
		assert(mic_dat != NULL);

		//log status string
		sprintf(message, "ctor ok");
	}


	bool open(PaDeviceIndex index)
	{
		PaStreamParameters inputParameters;
		PaStreamParameters outputParameters;

		inputParameters.device = index;
		outputParameters.device = index;

		if (outputParameters.device == paNoDevice)
			return false;
		if (inputParameters.device == paNoDevice)
			return false;

		const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(index);
		if (pInfo != 0)
			printf("Output device name: '%s'\r", pInfo->name);

		outputParameters.channelCount = (int)(spk_wav.GetNumChannel());         /* dependes on the wav file */
		outputParameters.sampleFormat = paFloat32;                              /* CsWav will alway ensure 32 bit float format */
		outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
		outputParameters.hostApiSpecificStreamInfo = NULL;

		inputParameters.channelCount = (int)(mic_channels);         /* dependes on the wav file */
		inputParameters.sampleFormat = paFloat32;                   /* CsWav will alway ensure 32 bit float format */
		inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowOutputLatency;
		inputParameters.hostApiSpecificStreamInfo = NULL;

		//frames per buffer can also be "paFramesPerBufferUnspecified"
		//Using 'this' for userData so we can cast to PaPlay* in paCallback method
		if (paNoError != Pa_OpenStream(&stream, &inputParameters, &outputParameters, sample_rate, FRAMES_PER_BUFFER, paClipOff, &PaPlayRecord::paCallback, this))
			return false;
		if (paNoError != Pa_SetStreamFinishedCallback(stream, &PaPlayRecord::paStreamFinished))
		{
			Pa_CloseStream(stream);
			stream = 0;
			return false;
		}
		return true;
	}


	bool info()
	{
		if (stream == 0)
			return false;
		const PaStreamInfo * stream_info = Pa_GetStreamInfo(stream);
		printf("PaStreamInfo: struct version = %d\n", stream_info->structVersion);
		printf("PaStreamInfo: input latency = %f second\n", stream_info->inputLatency);
		printf("PaStreamInfo: output latency = %f second\n", stream_info->outputLatency);
		printf("PaStreamInfo: sample rate = %f sps\n", stream_info->sampleRate);
		return paNoError;
	}


	bool close()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_CloseStream(stream);
		stream = 0;
		return (err == paNoError);
	}

	bool start()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_StartStream(stream);
		return (err == paNoError);
	}

	bool pending()
	{
		if (stream == 0)
			return false;
		printf("\n");
		while (Pa_IsStreamActive(stream))
		{
			printf("\rcpu load:[%f], patime[%f]", cpuload, timebase); fflush(stdout);
			Pa_Sleep(250);
		}
		return paNoError;
	}

	bool stop()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_StopStream(stream);
		return (err == paNoError);
	}

	size_t save()
	{
		size_t bytes_written = mic_wav.SaveFile(path_written.c_str(), bits_persample);
		return bytes_written;
	}


	volatile double cpuload;
	volatile PaTime timebase;

private:
	/* The instance callback, where we have access to every method/variable in object of class PaPlay */
	int paCallbackMethod(
		const void *inputBuffer,
		void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags)
	{
		const float *in = (const float *)inputBuffer;
		float *out = (float*)outputBuffer;
		unsigned long i;

		(void)timeInfo; /* Prevent unused variable warnings. */
		(void)statusFlags;


		for (i = 0; i < framesPerBuffer; i++)
		{
			memcpy(out, &spk_dat[spk_frame], spk_wav.GetNumChannel() * sizeof(float));
			memcpy(mic_dat, in, mic_channels * sizeof(float));

			out += ((int)(spk_wav.GetNumChannel()));
			spk_frame += (spk_wav.GetNumChannel());
			in += ((int)mic_channels);
			mic_dat += ((int)mic_channels);
			mic_cntframes += 1; //for sentinel and verification only

			if (spk_frame >= spk_totsps)
			{
				assert(mic_cntframes == mic_totframes);
				memset(out, 0, (framesPerBuffer - i - 1) * (spk_wav.GetNumChannel()) * sizeof(float));
				spk_frame = 0;
				mic_cntframes = 0;
				return paComplete;
			}
		}

		//update utility features
		cpuload = Pa_GetStreamCpuLoad(stream);
		timebase = Pa_GetStreamTime(stream);

		return paContinue;
	}

	/* This routine will be called by the PortAudio engine when audio is needed.
	** It may called at interrupt level on some machines so don't do anything
	** that could mess up the system like calling malloc() or free().
	*/
	static int paCallback(
		const void *inputBuffer,
		void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags,
		void *userData)
	{
		/* Here we cast userData to PaPlay* type so we can call the instance method paCallbackMethod, we can do that since
		we called Pa_OpenStream with 'this' for userData */
		return ((PaPlayRecord*)userData)->paCallbackMethod(inputBuffer, outputBuffer, framesPerBuffer, timeInfo, statusFlags);
	}




	void paStreamFinishedMethod()
	{
		printf("Stream Completed: %s\n", message);
	}

	/*
	* This routine is called by portaudio when playback is done.
	*/
	static void paStreamFinished(void* userData)
	{
		return ((PaPlayRecord*)userData)->paStreamFinishedMethod();
	}


	//record
	std::string path_written;
	size_t bits_persample;
	CsWav mic_wav;
	float *mic_dat;
	size_t mic_channels;
	size_t mic_cntframes;
	size_t mic_totframes;
	size_t sample_rate;

	//play
	CsWav spk_wav;
	const float *spk_dat;
	size_t spk_frame;
	size_t spk_totsps;

	PaStream *stream;
	char message[20];
};

class PaRecord
{
public:
	PaRecord(
		std::string path_to_record,
		size_t process_sample_rate,
		size_t channels,
		double recording_time,
		size_t bits_per_sample
	) : stream(0), mic_cntframes(0), mic_dat(NULL)
	{
		//init recording wav source
		bits_persample = bits_per_sample;
		path_written = path_to_record;
		mic_channels = channels;
		sample_rate = process_sample_rate;
		mic_totframes = (size_t)std::ceil(recording_time * (double)sample_rate);

		mic_dat = mic_wav.SetFrameMatrix(mic_totframes, mic_channels, sample_rate);
		assert(mic_dat != NULL);

		//log status string
		sprintf(message, "ctor ok");
	}


	bool open(PaDeviceIndex index)
	{
		PaStreamParameters inputParameters;
		inputParameters.device = index;
		if (inputParameters.device == paNoDevice)
			return false;
		const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(index);
		if (pInfo != 0)
			printf("Output device name: '%s'\r", pInfo->name);

		inputParameters.channelCount = (int)(mic_channels);         /* dependes on the wav file */
		inputParameters.sampleFormat = paFloat32;                   /* CsWav will alway ensure 32 bit float format */
		inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowOutputLatency;
		inputParameters.hostApiSpecificStreamInfo = NULL;

		//frames per buffer can also be "paFramesPerBufferUnspecified"
		//Using 'this' for userData so we can cast to PaPlay* in paCallback method
		if (paNoError != Pa_OpenStream(&stream, &inputParameters, NULL, sample_rate, FRAMES_PER_BUFFER, paClipOff, &PaRecord::paCallback, this))
			return false;
		if (paNoError != Pa_SetStreamFinishedCallback(stream, &PaRecord::paStreamFinished))
		{
			Pa_CloseStream(stream);
			stream = 0;
			return false;
		}
		return true;
	}


	bool info()
	{
		if (stream == 0)
			return false;
		const PaStreamInfo * stream_info = Pa_GetStreamInfo(stream);
		printf("PaStreamInfo: struct version = %d\n", stream_info->structVersion);
		printf("PaStreamInfo: input latency = %f second\n", stream_info->inputLatency);
		printf("PaStreamInfo: output latency = %f second\n", stream_info->outputLatency);
		printf("PaStreamInfo: sample rate = %f sps\n", stream_info->sampleRate);
		return paNoError;
	}


	bool close()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_CloseStream(stream);
		stream = 0;
		return (err == paNoError);
	}

	bool start()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_StartStream(stream);
		return (err == paNoError);
	}

	bool pending()
	{
		if (stream == 0)
			return false;
		printf("\n");
		while (Pa_IsStreamActive(stream))
		{
			printf("\rcpu load:[%f], patime[%f]", cpuload, timebase); fflush(stdout);
			Pa_Sleep(250);
		}
		return paNoError;
	}

	bool stop()
	{
		if (stream == 0)
			return false;
		PaError err = Pa_StopStream(stream);
		return (err == paNoError);
	}

	size_t save()
	{
		size_t bytes_written = mic_wav.SaveFile(path_written.c_str(), bits_persample);
		return bytes_written;
	}


	volatile double cpuload;
	volatile PaTime timebase;

private:
	/* The instance callback, where we have access to every method/variable in object of class PaPlay */
	int paCallbackMethod(
		const void *inputBuffer,
		void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags)
	{
		const float *in = (const float *)inputBuffer;
		unsigned long i;

		(void)timeInfo; /* Prevent unused variable warnings. */
		(void)statusFlags;
		(void)outputBuffer;

		for (i = 0; i < framesPerBuffer; i++)
		{
			memcpy(mic_dat, in, mic_channels * sizeof(float));
			
			in += (int)mic_channels;
			mic_dat += (int)mic_channels;
			mic_cntframes += 1;
			
			if (mic_cntframes >= mic_totframes)
			{
				mic_cntframes = 0;
				return paComplete;
			}
		}

		//update utility features
		cpuload = Pa_GetStreamCpuLoad(stream);
		timebase = Pa_GetStreamTime(stream);

		return paContinue;
	}

	/* This routine will be called by the PortAudio engine when audio is needed.
	** It may called at interrupt level on some machines so don't do anything
	** that could mess up the system like calling malloc() or free().
	*/
	static int paCallback(
		const void *inputBuffer,
		void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags,
		void *userData)
	{
		/* Here we cast userData to PaPlay* type so we can call the instance method paCallbackMethod, we can do that since
		we called Pa_OpenStream with 'this' for userData */
		return ((PaRecord*)userData)->paCallbackMethod(inputBuffer, outputBuffer, framesPerBuffer, timeInfo, statusFlags);
	}




	void paStreamFinishedMethod()
	{
		printf("Stream Completed: %s\n", message);
	}

	/*
	* This routine is called by portaudio when playback is done.
	*/
	static void paStreamFinished(void* userData)
	{
		return ((PaRecord*)userData)->paStreamFinishedMethod();
	}


	std::string path_written;
	size_t bits_persample;
	CsWav mic_wav;
	float *mic_dat;
	size_t mic_channels;
	size_t mic_cntframes;
	size_t mic_totframes;
	size_t sample_rate;

	PaStream *stream;
	char message[20];
};

class PaPlay
{
public:
    PaPlay(
		std::string path_to_play, 
		size_t process_sample_rate
		//int * out_channel_select,
		//int out_channels
		) : stream(0), spk_frame(0)
    {
		//init playback wav source
		size_t head_bytes = spk_wav.ExtractMetaInfo(path_to_play.c_str());
		spk_wav.PrintMetaInfo();
		assert(spk_wav.GetSampleRate() == process_sample_rate);
		sample_rate = process_sample_rate;
		spk_totsps = spk_wav.GetFrameLength() * spk_wav.GetNumChannel();

		//load the playback data from the wav
		spk_dat = spk_wav.GetFrameMatrix(path_to_play.c_str());

		//routing infomation
		//memcpy(outputChannelSelectors, out_channel_select, out_channels * sizeof(int));

		//log status string
        sprintf( message, "ctor ok" );
    }


    bool open(PaDeviceIndex index)
    {
        PaStreamParameters outputParameters;
        outputParameters.device = index;
        if (outputParameters.device == paNoDevice)
            return false;
        const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(index);
        if (pInfo != 0)
            printf("Output device name: '%s'\r", pInfo->name);

        outputParameters.channelCount = (int)(spk_wav.GetNumChannel());         /* dependes on the wav file */
        outputParameters.sampleFormat = paFloat32;                              /* CsWav will alway ensure 32 bit float format */
        outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = NULL;

		/* Use an ASIO specific structure. WARNING - this is not portable. */
		//asioOutputInfo.size = sizeof(PaAsioStreamInfo);
		//asioOutputInfo.hostApiType = paASIO;
		//asioOutputInfo.version = 1;
		//asioOutputInfo.flags = paAsioUseChannelSelectors;
		////outputChannelSelectors[0] = 1; /* skip channel 0 and use the second (right) ASIO device channel */
		//asioOutputInfo.channelSelectors = outputChannelSelectors;
		//outputParameters.hostApiSpecificStreamInfo = &asioOutputInfo;


		//frames per buffer can also be "paFramesPerBufferUnspecified"
		//Using 'this' for userData so we can cast to PaPlay* in paCallback method
        if (paNoError != Pa_OpenStream(&stream, NULL, &outputParameters, sample_rate, FRAMES_PER_BUFFER, paClipOff, &PaPlay::paCallback, this))
            return false;
        if (paNoError != Pa_SetStreamFinishedCallback(stream, &PaPlay::paStreamFinished))
        {
            Pa_CloseStream( stream );
            stream = 0;
            return false;
        }
        return true;
    }


	bool info()
	{
		if (stream == 0)
			return false;
		const PaStreamInfo * stream_info = Pa_GetStreamInfo(stream);
		printf("PaStreamInfo: struct version = %d\n", stream_info->structVersion);
		printf("PaStreamInfo: input latency = %f second\n", stream_info->inputLatency);
		printf("PaStreamInfo: output latency = %f second\n", stream_info->outputLatency);
		printf("PaStreamInfo: sample rate = %f sps\n", stream_info->sampleRate);
		return paNoError;
	}


    bool close()
    {
        if (stream == 0)
            return false;
        PaError err = Pa_CloseStream( stream );
        stream = 0;
        return (err == paNoError);
    }

    bool start()
    {
        if (stream == 0)
            return false;
        PaError err = Pa_StartStream( stream );
        return (err == paNoError);
    }

	bool pending()
	{
		if (stream == 0)
			return false;
		printf("\n");
		while (Pa_IsStreamActive(stream))
		{
			printf("\rcpu load:[%f], patime[%f]", cpuload, timebase); fflush(stdout);
			Pa_Sleep(250);
		}
		return paNoError;
	}

    bool stop()
    {
        if (stream == 0)
            return false;
        PaError err = Pa_StopStream( stream );
        return (err == paNoError);
    }


	volatile double cpuload;
	volatile PaTime timebase;

private:
    /* The instance callback, where we have access to every method/variable in object of class PaPlay */
    int paCallbackMethod(
		const void *inputBuffer, 
		void *outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags)
    {
        float *out = (float*)outputBuffer;
        unsigned long i;

        (void) timeInfo; /* Prevent unused variable warnings. */
        (void) statusFlags;
        (void) inputBuffer;

        for( i = 0; i < framesPerBuffer; i++ )
        {
			memcpy(out, &spk_dat[spk_frame], spk_wav.GetNumChannel() * sizeof(float));
			out += ((int)(spk_wav.GetNumChannel()));
			spk_frame += (spk_wav.GetNumChannel());
			if (spk_frame >= spk_totsps)
			{
				memset(out, 0, (framesPerBuffer-i-1) * (spk_wav.GetNumChannel()) * sizeof(float));
				spk_frame = 0;
				return paComplete;
			}
        }

		//update utility features
		cpuload = Pa_GetStreamCpuLoad(stream);
		timebase = Pa_GetStreamTime(stream);

        return paContinue;
    }

    /* This routine will be called by the PortAudio engine when audio is needed.
    ** It may called at interrupt level on some machines so don't do anything
    ** that could mess up the system like calling malloc() or free().
    */
    static int paCallback( 
		const void *inputBuffer, 
		void *outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData )
    {
        /* Here we cast userData to PaPlay* type so we can call the instance method paCallbackMethod, we can do that since 
           we called Pa_OpenStream with 'this' for userData */
        return ((PaPlay*)userData)->paCallbackMethod(inputBuffer, outputBuffer, framesPerBuffer, timeInfo, statusFlags);
    }


    void paStreamFinishedMethod()
    {
        printf( "Stream Completed: %s\n", message );
    }

    /*
     * This routine is called by portaudio when playback is done.
     */
    static void paStreamFinished(void* userData)
    {
        return ((PaPlay*)userData)->paStreamFinishedMethod();
    }


	CsWav spk_wav;
	const float *spk_dat;
	size_t spk_frame;
	size_t spk_totsps;

	//PaAsioStreamInfo asioOutputInfo;
	//int outputChannelSelectors[MAX_SPEAKER_IO];

	size_t sample_rate;
	PaStream *stream;
    char message[20];
};

class ScopedPaHandler
{
public:
    ScopedPaHandler()
        : _result(Pa_Initialize())
    {
    }
    ~ScopedPaHandler()
    {
        if (_result == paNoError)
            Pa_Terminate();
    }

    PaError result() const { return _result; }

private:
    PaError _result;
};








/*******************************************************************/
//int main(void);
//int main(void)
int playrecord(const char * path_play, const char * path_record, int sample_rate, int channels_record, int bits_record)
{
	printf("PortAudio Test: I/O PaPlay and PaRecord. SR = %d, BufSize = %d\n", 48000, FRAMES_PER_BUFFER);
	
	std::string spath(path_play);
	std::string rpath(path_record);
	
	PaPlayRecord PaPlayRecord(spath, rpath, sample_rate, channels_record, bits_record);
	ScopedPaHandler paInit;

	if (paInit.result() != paNoError)
		goto error;

	if (PaPlayRecord.open(Pa_GetDefaultOutputDevice()))
	{
		PaPlayRecord.info();
		if (PaPlayRecord.start())
		{
			PaPlayRecord.pending();
			PaPlayRecord.stop();
			PaPlayRecord.save();
		}
		PaPlayRecord.close();
	}
	return paNoError;

error:
	fprintf(stderr, "An error occured while using the portaudio stream\n");
	fprintf(stderr, "Error number: %d\n", paInit.result());
	fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(paInit.result()));
	return 1;
}

int play(const char * path, int sample_rate)
{
	printf("PortAudio Test: output PaPlay wave. SR = %d, BufSize = %d\n", 48000, FRAMES_PER_BUFFER);
    //PaPlay PaPlay("D:\\pa_stable_v190600_20161030\\PaDynamic\\8_Channel_ID.wav", 48000);
	std::string spath(path);
	PaPlay PaPlay(spath, sample_rate);

	ScopedPaHandler paInit;
    if( paInit.result() != paNoError ) 
		goto error;

    if (PaPlay.open(Pa_GetDefaultOutputDevice()))
    {
		PaPlay.info();
        if (PaPlay.start())
        {    
			PaPlay.pending();
            PaPlay.stop();
        }
        PaPlay.close();
    }
    return paNoError;

error:
    fprintf( stderr, "An error occured while using the portaudio stream\n" );
    fprintf( stderr, "Error number: %d\n", paInit.result() );
    fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( paInit.result() ) );
    return 1;
}

int record(const char * path, int sample_rate, int channels, double duration, int bits)
{
	printf("PortAudio Test: input PaRecord wave. SR = %d, BufSize = %d\n", 48000, FRAMES_PER_BUFFER);
	std::string rpath(path);
	PaRecord PaRecord(rpath, sample_rate, channels, duration, bits);

	ScopedPaHandler paInit;
	if (paInit.result() != paNoError)
		goto error;

	if (PaRecord.open(Pa_GetDefaultOutputDevice()))
	{
		PaRecord.info();
		if (PaRecord.start())
		{
			PaRecord.pending();
			PaRecord.stop();
			PaRecord.save();
		}
		PaRecord.close();
	}
	return paNoError;

error:
	fprintf(stderr, "An error occured while using the portaudio stream\n");
	fprintf(stderr, "Error number: %d\n", paInit.result());
	fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(paInit.result()));
	return 1;
}