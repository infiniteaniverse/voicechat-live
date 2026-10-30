// AudioWorkletProcessor that batches raw mic samples into ~32ms Float32 chunks
// and posts them to the main thread. Runs off the main thread (unlike the
// deprecated ScriptProcessorNode), so mic capture can't glitch a recording.
// Implementation: Part 8 (stretch goal)

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const frameMs = options.processorOptions?.frameMs ?? 32;
    this._targetSamples = Math.round((sampleRate * frameMs) / 1000);
    this._buffer = new Float32Array(this._targetSamples);
    this._offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channel = input[0]; // mono
    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._offset++] = channel[i];
      if (this._offset === this._targetSamples) {
        this.port.postMessage(this._buffer.slice(0));
        this._offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
