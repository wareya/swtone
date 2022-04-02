using Microsoft.CSharp;
using System;
using Godot;
using Dictionary = Godot.Collections.Dictionary;

using System.Collections.Generic;
using Array = System.Collections.ArrayList;


public class Main : Control
{
     
    public class Filter : Reference
    {
        public float sample_rate = 1.0f;
        public Filter(float sample_rate)
        {
            this.sample_rate = sample_rate;
        }
    
        public virtual void PushSample(Vector2 x){}
        public virtual Vector2 PopSample(){return new Vector2();}
    }
    
    public class Delay : Filter
    {
        public float wet_amount = 0.25f;
        public float dry_amount = 1.0f;
        public float decay = 0.2f;
        public float stereo_offset = -0.01f;
        public int diffusion = 8;
        public float diffusion_ratio = 1.0f;
        
        public List<Vector2> wet;
        public Vector2 dry = Vector2.Zero;
        public int cursor = 0;
        
        public Delay(float sample_rate, float time, float decay) : base(sample_rate)
        {
            this.decay = decay;
            wet = new List<Vector2>();
            foreach(var _i in GD.Range((int)(time*sample_rate)))
            {
                wet.Add(Vector2.Zero);
            }
        }
        
        public override void PushSample(Vector2 x)
        {
            wet[cursor] = (Vector2)wet[cursor] * decay;
            wet[cursor] = (Vector2)wet[cursor] + x;
            dry = x;
            cursor = (cursor + 1) % wet.Count;
        
        }
    
        public Vector2 Sample(int c)
        {
            return (Vector2)wet[(c % wet.Count + wet.Count ) % wet.Count];
        
        }
    
        public Vector2 PullFromCursor(int c)
        {
            var _wet = Sample(c);
            var offset = (int)Mathf.Floor(Mathf.Abs(stereo_offset * wet.Count));
            if(diffusion > 0)
            {
                offset = (int)Mathf.Floor(offset / (diffusion + 1.0f));
            }
            if(stereo_offset > 0.0f)
            {
                _wet.y = Sample(c+offset).y;
            }
            else if(stereo_offset < 0.0f)
            {
                _wet.x = Sample(c+offset).x;
            }
            return _wet;
            
        }
    
        public override Vector2 PopSample()
        {
            if(diffusion == 0)
            {
                return dry * dry_amount + PullFromCursor(cursor) * wet_amount;
            }
            else
            {
                Vector2 _wet = new Vector2();
                foreach(var i in GD.Range(diffusion+1))
                {
                    float d = (float)(i)/(float)(diffusion+1);
                    GD.Seed((ulong)i ^ (ulong)GD.Hash(wet.Count ^ 588123));
                    d += (float)GD.RandRange(-0.5f, 0.5f)/(float)(diffusion+1);
                    d *= diffusion_ratio;
                    int d2 = (int)(d*wet.Count);
                    _wet += PullFromCursor(cursor + d2);
                }
                _wet = _wet/(float)(diffusion+1);
                return dry*dry_amount + _wet*wet_amount;
            }
        }
    }
    
    public class Reverb : Filter
    {
        public float dry_amount = 1.0f;
        public float wet_amount = 1.0f;
        public float decay = 0.5f;
        
        public Delay delay1;
        public Delay delay2;
        public Delay delay3;
        public Lowpass lowpass;
        public Vector2 dry = new Vector2(){};
        
        public Reverb(float sample_rate, float _decay) : base(sample_rate)
        {
            decay = _decay;
            delay1 = new Delay(sample_rate, 0.19f, decay);
            delay1.wet_amount = 1.0f;
            delay1.dry_amount = 1.0f/8.0f;
            delay1.diffusion = 8;
            delay1.diffusion_ratio = 1.0f;
            
            delay2 = new Delay(sample_rate, 0.14f, 0.0f);
            delay2.wet_amount = 1.0f;
            delay2.dry_amount = 1.0f/8.0f;
            delay2.diffusion = 8;
            delay2.stereo_offset = 0.02f;
            delay2.diffusion_ratio = 1.0f;
            
            delay3 = new Delay(sample_rate, 0.031f, 0.0f);
            delay3.wet_amount = 1.0f;
            delay3.dry_amount = 1.0f/8.0f;
            delay3.diffusion = 8;
            delay2.stereo_offset = 0.5f;
            delay3.diffusion_ratio = 1.0f;
            
            lowpass = new Lowpass(sample_rate, 2000.0f);
        }
        
        public override void PushSample(Vector2 x)
        {
            dry = x;
        }
    
        public override Vector2 PopSample()
        {
            delay1.PushSample(dry);
            delay2.PushSample(delay1.PopSample());
            delay3.PushSample(delay2.PopSample());
            lowpass.PushSample(delay3.PopSample());
            return dry * dry_amount + lowpass.PopSample() * wet_amount * 8.0f;
        }
    }
    
    public class Lowpass : Filter
    {
        public float cutoff = 0.0f;
        public float decay_constant = 0.0f;
        public Vector2 memory = new Vector2(){};
        public Lowpass(float sample_rate, float cutoff) : base(sample_rate)
        {
            UpdateDecayConstant(cutoff);
        }
        public void UpdateDecayConstant(float cutoff)
        {
            this.cutoff = cutoff;
            float y = 1.0f - Mathf.Cos(cutoff / (sample_rate/2.0f) * Mathf.Pi);
            decay_constant = -y + Mathf.Sqrt(y*y + 2.0f*y);
        }
    
        public override void PushSample(Vector2 x)
        {
            memory = memory.LinearInterpolate(x, decay_constant);
        }
    
        public override Vector2 PopSample()
        {
            return memory;
        }
    }
    
    public class Highpass : Filter
    {
        public float cutoff = 0.0f;
        public float decay_constant = 0.0f;
        public Vector2 memory = new Vector2(){};
        public Vector2 memory2 = new Vector2(){};
        public Highpass(float sample_rate, float cutoff) : base(sample_rate)
        {
            UpdateDecayConstant(cutoff);
        }
        public void UpdateDecayConstant(float cutoff)
        {
            this.cutoff = cutoff;
            float y = 1.0f - Mathf.Cos(cutoff / (sample_rate/2.0f) * Mathf.Pi);
            decay_constant = -y + Mathf.Sqrt(y*y + 2.0f*y);
        }
    
        public override void PushSample(Vector2 x)
        {
            memory = memory.LinearInterpolate(x, decay_constant);
            memory2 = x;
        }
    
        public override Vector2 PopSample()
        {
            return memory2 - memory;
        }
    }
    public class Ringmod : Filter
    {
        public float frequency = 0.0f;
        public float amount = 1.0f;
        public float cursor = 0.0f;
        public float phase = 0.0f;
        public Vector2 memory = new Vector2(){};
        public Ringmod(float sample_rate, float frequency, float phase, float amount) : base(sample_rate)
        {
            this.frequency = frequency;
            this.amount = amount;
            this.phase = phase;
            cursor = 0.0f;
        }
        public override void PushSample(Vector2 x)
        {
            memory = x;
        }
    
        public override Vector2 PopSample()
        {
            var r = memory * Mathf.Sin(cursor*Mathf.Pi*2.0f + phase*Mathf.Pi*2.0f);
            cursor += frequency*2.0f/sample_rate;
            return memory.LinearInterpolate(r, amount);
        }
    }
    
    public class Flanger : Filter
    {
        public int min_depth = 0;
        public int max_depth = 0;
        public float rate = 2.0f;
        public float wet_amount = 1.0f;
        public float dry_amount = 1.0f;
        public float feedback = 0.5f;
        
        public int cursor = 0;
        public List<Vector2> wet;
        
        public Flanger(float sample_rate, float _min_depth, float _max_depth, float _rate) : base(sample_rate)
        {
            wet = new List<Vector2>();
            this.min_depth = (int)(_min_depth * sample_rate);
            this.max_depth = (int)(_max_depth * sample_rate);
            this.rate = (float)(_rate);
            float count = Mathf.Max(min_depth, max_depth);
            foreach(var _i in GD.Range((int)(Mathf.Ceil(count))))
            {
                wet.Add(Vector2.Zero);
            }
            if(wet.Count == 0)
            {
                wet.Add(Vector2.Zero);
            }
        }
        public override void PushSample(Vector2 x)
        {
            wet[cursor % wet.Count] = x;
            cursor += 1;
        }
    
        public float _Tri(float x)
        {
            return Mathf.Abs((Mathf.PosMod((x*2.0f)+1.0f,2.0f)-1.0f));
        }
        
        public override Vector2 PopSample()
        {
            var lfo_amount = _Tri(cursor/sample_rate * (rate > 0.0f ? (1.0f/rate) : 0.0f));
            float _samples_into_past = Mathf.Lerp(min_depth, max_depth, lfo_amount);
            int samples_into_past = (int)(Mathf.Min(_samples_into_past, wet.Count-1));
            //if GD.Randi() % 999 == 0:
            //    GD.Print("%s/%s" % [samples_into_past, wet.Count])
            var _wet = wet[(cursor - 1 - samples_into_past + wet.Count*2) % wet.Count];
            var c = (cursor - 1) % wet.Count;
            var dry =  wet[c];
            var output = dry * dry_amount + _wet * wet_amount;
            wet[c] *= 1.0f - feedback;
            wet[c] += output * feedback;
            return output;
            // TODO add stereo thing somehow
        }
    }
    
    public class Chorus : Filter
    {
        //int min_depth = 0;
        public float depth = 0.0f;
        public float rate = 2.0f;
        public float wet_amount = 1.0f;
        public float dry_amount = 0.0f;
        public int voices = 3;
        
        public int cursor = 0;
        public List<Vector2> wet;
        //var stereo_offset = -0.01;
        public Chorus(float sample_rate, float depth, float _rate, float _wet, float _dry, int _voices) : base(sample_rate)
        {
            rate = (float)(_rate);
            wet_amount = _wet;
            dry_amount = _dry;
            voices = _voices;
            
            wet = new List<Vector2>();
            foreach(var _i in GD.Range((int)(depth*sample_rate)))
            {
                wet.Add(Vector2.Zero);
            }
            if(wet.Count == 0)
            {
                wet.Add(Vector2.Zero);
        
            }
        }
        public override void PushSample(Vector2 x)
        {
            wet[cursor % wet.Count] = x;
            cursor += 1;
        }
    
        public float _Tri(float x)
        {
            return Mathf.Abs((Mathf.PosMod((x*2.0f)+1.0f,2.0f)-1.0f));
        }
    
        public override Vector2 PopSample()
        {
            Vector2 lfo_sum = new Vector2();
            foreach(var i in GD.Range(voices))
            {
                var lfo_amount = -Mathf.Cos((cursor*rate/sample_rate + 2.0f*i/voices) * Mathf.Pi)/2.0f + 0.5f;
                var samples_into_past = (int)((wet.Count-1) * lfo_amount);
                //if GD.Randi() % 999 == 0 && i == 0:
                //    GD.Print("%s/%s" % [samples_into_past, wet.Count])
                var _wet = wet[(cursor - 1 - samples_into_past + wet.Count*2) % wet.Count];
                lfo_sum += _wet;
            }
            var c = (cursor - 1) % wet.Count;
            var dry =  wet[c];
            var output = dry * dry_amount + lfo_sum * wet_amount;
            return output;
            // TODO: add optional reference low pass
        }
    }
    
    public class Waveshaper : Filter
    {
        public float pre_gain = 1.0f;
        public float exponent = 1.0f;
        public float clip = 1.0f;
        public int clip_mode = 0; // 0: normal clipping. 1: "bounce" clipping. 2: wrapping
        public int quantization = 0; // quantize to number of steps
        public float mix = 0.0f;
        
        public Vector2 memory = new Vector2();
        public Waveshaper(float sample_rate, float pre_gain, float exponent, float clip, int clip_mode, int quantization, float mix) : base(sample_rate)
        {
            this.pre_gain = pre_gain;
            this.exponent = exponent;
            this.clip = clip;
            this.clip_mode = clip_mode;
            this.quantization = quantization;
            this.mix = mix;
        }
        public override void PushSample(Vector2 x)
        {
            memory = x;
        }
    
        public float _Saw(float x)
        {
            var r = Mathf.PosMod(Mathf.PosMod(x+1.0f, 2.0f) + 2.0f, 2.0f) - 1.0f;
            return r;
        }
    
        public float _Tri(float x)
        {
            return Mathf.Abs(Mathf.PosMod(Mathf.PosMod(x-1.0f, 4.0f) + 4.0f, 4.0f) - 2.0f) - 1.0f;
        }
    
        public override Vector2 PopSample()
        {
            Vector2 shaped = memory*pre_gain;
            shaped = shaped.Abs();
            shaped.x = Mathf.Pow(shaped.x, exponent);
            shaped.y = Mathf.Pow(shaped.y, exponent);
            shaped = memory.Sign() * shaped;
            if(clip > 0.0)
            {
                if(clip_mode == 0)
                {
                    shaped.x = Mathf.Clamp(shaped.x, -clip, clip);
                    shaped.y = Mathf.Clamp(shaped.y, -clip, clip);
                }
                else if(clip_mode == 1)
                {
                    shaped.x = _Tri(shaped.x/clip) * clip;
                    shaped.y = _Tri(shaped.y/clip) * clip;
                }
                else if(clip_mode == 2)
                {
                    shaped.x = _Saw(shaped.x/clip) * clip;
                    shaped.y = _Saw(shaped.y/clip) * clip;
                }
            }
            else
            {
                shaped = new Vector2();
            }
            var stages = quantization/2.0f - 0.5f;
            if(stages > 0.0f)
            {
                shaped.x = Mathf.Floor(shaped.x*stages)/stages + 1.0f/stages/2.0f;
                shaped.y = Mathf.Floor(shaped.y*stages)/stages + 1.0f/stages/2.0f;
            }
            return memory.LinearInterpolate(shaped, mix);
    
        }
    }
    
    public class Limiter : Filter
    {
        public float pre_gain = 1.0f;
        public float lookahead = 0.001f;
        public float attack = 0.001f;
        public float sustain = 0.04f;
        public float release = 0.04f;
        public float threshold = 1.0f;
        
        public float amplitude = 1.0f;
        public float hit_time = 0.0f;
        public float hit_amplitude = 1.0f;
        public float time = 0.0f;
        
        public List<Vector2> buffer;
        public List<float> buffer_max;
        public List<float> buffer_max_bucket;
        
        public int bucket_dirty = -1;
        public int bucket_size = 128;
        public int cursor = 0;
        
        public float attack_amplitude_raw = 1.0f;
        public float attack_amplitude = 1.0f;
        public List<float> attack_buffer;
        public int attack_cursor = 0;
        // FIXME: identify whatever bug is causing this: https://i.imgur.com/eji72dw.png
        // lookahead? attack? no clue
        public Limiter(float sample_rate, float _pre_gain, float _lookahead, float _attack, float _sustain, float _release, float _threshold) : base(sample_rate)
        {
            pre_gain = _pre_gain;
            lookahead = _lookahead;
            attack = _attack;
            sustain = _sustain;
            release = _release;
            threshold = _threshold;
            
            bucket_size = Mathf.Max(1, (int)(Mathf.Sqrt((sustain + lookahead) * sample_rate)));
            GD.Print($"bucket size {bucket_size}");
            
            buffer = new List<Vector2>();
            buffer_max = new List<float>();
            buffer_max_bucket = new List<float>();
            
            attack_buffer = new List<float>();
            
            foreach(var _i in GD.Range((int)(lookahead * sample_rate + 1)))
            {
                buffer.Add(new Vector2());
            }
            foreach(var i in GD.Range((int)((sustain + lookahead) * sample_rate)))
            {
                buffer_max.Add(0.0f);
                if((int)(i) % bucket_size == 0)
                {
                    buffer_max_bucket.Add(0.0f);
                }
            }
            if(buffer.Count == 0)
            {
                buffer.Add(new Vector2());
            }
            if(buffer_max.Count == 0)
            {
                buffer_max.Add(0.0f);
                buffer_max_bucket.Add(0.0f);
            }
            time = 0.0f;
        }
        
        float last_max = 0.0f;
        public float MaxBuffer()
        {
            if(bucket_dirty >= 0)
            {
                float y = 0.0f;
                //foreach(var j in GD.Range(bucket_dirty*bucket_size, Mathf.Min((bucket_dirty+1)*bucket_size, buffer_max.Count)))
                var max = Mathf.Min((bucket_dirty+1)*bucket_size, buffer_max.Count);
                for(var j = bucket_dirty*bucket_size; j < max; j++)
                {
                    var n = buffer_max[j];
                    y = Mathf.Max(n, y);
                }
                buffer_max_bucket[bucket_dirty] = y;
                bucket_dirty = -1;
            }
            float x = 0.0f;
            for(var n = 0; n < buffer_max_bucket.Count; n++)
            {
                x = Mathf.Max(buffer_max_bucket[n], x);
            }
            last_max = x;
            return x;
        }
    
        public float former_amplitude = 1.0f;
        public override void PushSample(Vector2 x)
        {
            if(threshold == 0.0)
            {
                return;
            }
            cursor += 1;
            x *= pre_gain;
            time += 1.0f/sample_rate;
            
            buffer[cursor % buffer.Count] = x;
            var loudness = Mathf.Max(Mathf.Abs(x.x), Mathf.Abs(x.y));
            var old_loudness = buffer_max[cursor % buffer_max.Count];
            buffer_max[cursor % buffer_max.Count] = loudness;
            
            var bucket_index = (cursor % buffer_max.Count)/bucket_size;
            var cached_for_bucket = buffer_max_bucket[bucket_index];
            float envelope = last_max;
            
            if (old_loudness >= last_max || loudness > last_max)
            {
                bucket_dirty = bucket_index;
                envelope = MaxBuffer();
            }
            else if (old_loudness >= cached_for_bucket || loudness > cached_for_bucket)
            {
                bucket_dirty = bucket_index;
                envelope = MaxBuffer(); // value will be the same but still need to call it
            }
            
            if(release > 0)
            {
                var time_since_hit = time - hit_time;
                if(time_since_hit < release)
                {
                    var decay_progress = time_since_hit/release;
                    amplitude = Mathf.Lerp(hit_amplitude, 1.0f, decay_progress);
                }
                else
                {
                    amplitude = 1.0f;
                    
                }
                var amplitude_follower = threshold/Mathf.Max(envelope, threshold);
                if(amplitude_follower <= amplitude)
                {
                    hit_time = time;
                    hit_amplitude = amplitude_follower;
                    amplitude = hit_amplitude;
                }
            }
            else
            {
                amplitude = threshold/Mathf.Max(envelope, threshold);
            }
            HandleAttack();
        }
        
        public void HandleAttack()
        {
            if(attack == 0.0f)
            {
                attack_amplitude = amplitude;
                return;
            }
            if(attack_buffer.Count == 0)
            {
                attack_amplitude = 0.0f;
                foreach(var _i in GD.Range((int)(attack * sample_rate + 1)))
                {
                    attack_buffer.Add(amplitude);
                    attack_amplitude += amplitude;
                }
                attack_amplitude /= (float)attack_buffer.Count;
            }
            var old = attack_buffer[attack_cursor % attack_buffer.Count];
            attack_buffer[attack_cursor % attack_buffer.Count] = amplitude;
            attack_cursor += 1;
            
            // this "attack" algorithm is numerically unstable but very fast
            // and the instability doesn't matter for the mere seconds/minutes of audio swtone produces
            attack_amplitude += amplitude/(float)attack_buffer.Count - old/(float)attack_buffer.Count;
            /*
            attack_amplitude = 0.0f;
            foreach(var b in attack_buffer)
            {
                attack_amplitude += b / attack_buffer.Count;
            }
            */
        }
    
        public override Vector2 PopSample()
        {
            if(threshold == 0.0f)
            {
                return new Vector2();
            }
            if(lookahead > 0.0f)
            {
                return buffer[(cursor + 1) % buffer.Count] * attack_amplitude;
            }
            else
            {
                return buffer[cursor % buffer.Count] * attack_amplitude;
            }
        }
    }
    
    public class Generator : Reference
    {
        public Node parent;
        public List<Vector2> samples;
        public float sample_rate = 44100.0f;
        public float freq = 440.0f;
        public float freq_offset_lfo = 0.0f; // semitones
        public float freq_offset_sweep = 0.0f; // semitones
        public float freq_offset_step = 0.0f; // semitones
        
        public float SemitonesToFactor(float x)
        {
            return Mathf.Pow(2.0f, x/12.0f);
        }
    
        public float FactorToSemitones(float x)
        {
            return Mathf.Log(x)/Mathf.Log(2.0f)*12.0f;
        }
    
        public int oversampling = 1;
        
        public float gen_time = 0.0f;
        public float gen_cursor = 0.0f;
        public float _Sin(float cursor)
        {
            return Mathf.Sin(cursor*Mathf.Pi);
        
        }
    
        public float _Tri(float cursor, float stages = 0.0f)
        {
            stages = stages/2.0f - 0.5f;
            var x = cursor;
            x = Mathf.Abs((Mathf.PosMod(x+1.5f,2.0f)-1.0f))*2.0f-1.0f;
            if(stages <= 0.0f)
            {
                return x;
            }
            else
            {
                return Mathf.Floor(x*stages)/stages + 1.0f/stages/2.0f;
        
            }
        }
    
        public float last_sq_cursor = 0.0f;
        public float _Square(float cursor, float width = 0.5f, bool nooffset = false)
        {
            var x = Mathf.PosMod(cursor, 2.0f)/2.0f;
            var output = x < width ? -1.0f : 1.0f;
            
            // FIXME make this work even when the cursor is cycling "backwards"
            if(!nooffset)
            {
                if(last_sq_cursor < width && x >= width)
                {
                    var between = Mathf.InverseLerp(x, last_sq_cursor, width);
                    output = Mathf.Lerp(-1.0f, 1.0f, between);
                }
                else if(x < last_sq_cursor)
                {
                    var between = Mathf.InverseLerp(x + 1.0f, last_sq_cursor, 1.0f);
                    output = Mathf.Lerp(1.0f, -1.0f, between);
                
                }
            }
            var dc_bias = (width - 0.5f) * 2.0f;
            if(nooffset)
            {
                dc_bias = 0.0f;
            }
            if(!nooffset)
            {
                last_sq_cursor = x;
            }
            return output + dc_bias;
        
        }
    
        public float last_saw_cursor = 0.0f;
        public float _Saw(float cursor, float exponent = 1.0f, bool no_magic = false)
        {
            var n = Mathf.PosMod(cursor, 2.0f)-1.0f;
            var output = Mathf.Pow(Mathf.Abs(n), exponent)*Mathf.Sign(n);
            // FIXME make work properly when cursor is moving "backwards"
            if(!no_magic)
            {
                if(n < last_saw_cursor && (last_saw_cursor - n) >= 1.0f)
                {
                    output = Mathf.Lerp(-1.0f, 1.0f, Mathf.InverseLerp(last_saw_cursor, n + 2.0f, 1.0f));
                }
                last_saw_cursor = n;
            }
            return output;
        
        }
    
        static public List<Vector2> MakePcmSource(AudioStreamSample clip)
        {
            int bytes_per_sample = 1;
            if(clip.Format == AudioStreamSample.FormatEnum.Format16Bits)
            {
                bytes_per_sample *= 2;
            }
            if(clip.Stereo)
            {
                bytes_per_sample *= 2;
            }
            var sample_count = clip.Data.Length / bytes_per_sample;
            var stream = new StreamPeerBuffer();
            stream.PutData(clip.Data);
            stream.Seek(0);
            var samples = new List<Vector2>();
            if(clip.Format == AudioStreamSample.FormatEnum.Format16Bits)
            {
                foreach(var _i in GD.Range(sample_count))
                {
                    var l = (float)(stream.Get16())/32768.0f;
                    var r = clip.Stereo ? (float)(stream.Get16())/32768.0f : l;
                    Vector2 sample = new Vector2(l, r);
                    samples.Add(sample);
                }
            }
            return samples;
        }
    
        public List<List<Vector2>> pcm_sources = new List<List<Vector2>>{
            MakePcmSource(GD.Load("res://paper bag.wav") as AudioStreamSample),
            MakePcmSource(GD.Load("res://plastic crinkle.wav") as AudioStreamSample),
            MakePcmSource(GD.Load("res://plastic crunch.wav") as AudioStreamSample),
            MakePcmSource(GD.Load("res://tambourine.wav") as AudioStreamSample),
        };
        public List<Vector2> pcm_source_custom;
        public float pcm_sample_loop = 1.0f;
        public float pcm_source = 0.0f;
        public float pcm_volume = 0.0f;
        public float pcm_offset = 0.0f;
        public float pcm_cutoff = 0.0f;
        public float pcm_rate = 16.0f;
        public int pcm_noise_cycle = 1024;
        public Vector2 _Pcm(float _cursor)
        {
            //if pcm_source == null || pcm_source.Count == 0:
            int cursor = (int)(_cursor*pcm_rate + pcm_offset*sample_rate);
            if((int)(pcm_source) == 0)
            {
                GD.Seed((ulong)(cursor % (int)(pcm_noise_cycle)));
                var n = GD.Randf() * 2.0f - 1.0f;
                return Vector2.One * n;
            }
            else if((int)(pcm_source) <= pcm_sources.Count)
            {
                var source = pcm_sources[(int)(pcm_source)-1];
                var size = source.Count;
                if(pcm_cutoff > 0.0f)
                {
                    size = Mathf.Min(size, (int)(pcm_cutoff * sample_rate));
                }
                if(cursor >= size)
                {
                    if(pcm_sample_loop != 0.0f)
                    {
                        cursor = cursor % size;
                    }
                    else
                    {
                        return Vector2.Zero;
                    }
                }
                return source[cursor];
            }
            else if((int)(pcm_source) == pcm_sources.Count+1)
            {
                var source = pcm_source_custom;
                if(source == null)
                {
                    return Vector2.Zero;
                }
                var size = source.Count;
                if(size == 0)
                {
                    return Vector2.Zero;
                }
                if(pcm_cutoff > 0.0f)
                {
                    size = Mathf.Min(size, (int)(pcm_cutoff * sample_rate));
                }
                if(cursor >= size)
                {
                    if(pcm_sample_loop != 0.0f)
                    {
                        cursor = cursor % size;
                    }
                    else
                    {
                        return Vector2.Zero;
                    }
                }
                return source[cursor];
            }
            else
            {
                return Vector2.Zero;
        
            }
        }
    
        public float time_limit = 5.0f;
        
        public float sin_volume = 0.0f;
        public float sin_detune = 0.0f;
        
        public float tri_volume = 0.0f;
        public float tri_stages = 16.0f;
        public float tri_detune = 0.0f;
        
        public float square_volume = 0.5f;
        public float square_width = 0.5f;
        public float square_detune = 0.0f;
        
        public float saw_volume = 0.0f;
        public float saw_exponent = 1.0f;
        public float saw_detune = 0.0f;
        
        public void UpdateFilters()
        {
            var c = samples.Count-1;
            
            if(delay_wet_amount != 0.0)
            {
                delay.PushSample(samples[c]);
                samples[c] = delay.PopSample();
            }
            else if(delay_dry_amount != 1.0)
            {
                samples[c] *= delay_dry_amount;
            }
            if(reverb_wet_amount != 0.0)
            {
                reverb.PushSample(samples[c]);
                samples[c] = reverb.PopSample();
            }
            else if(reverb_dry_amount != 1.0)
            {
                samples[c] *= reverb_dry_amount;
            }
            if(lowpass_frequency < 22050.0)
            {
                lowpass.PushSample(samples[c]);
                samples[c] = lowpass.PopSample();
            
            }
            if(highpass_frequency > 20.0)
            {
                highpass.PushSample(samples[c]);
                samples[c] = highpass.PopSample();
            
            }
            if(ringmod_frequency > 0.0 && ringmod_amount != 0.0)
            {
                ringmod.PushSample(samples[c]);
                samples[c] = ringmod.PopSample();
            
            }
            if(flanger_wet_amount != 0.0)
            {
                flanger.PushSample(samples[c]);
                samples[c] = flanger.PopSample();
            }
            else if(flanger_dry_amount != 1.0)
            {
                samples[c] *= flanger_dry_amount;
            }
            if((chorus_wet_amount != 0.0 && chorus_voices > 0))
            {
                chorus.PushSample(samples[c]);
                samples[c] = chorus.PopSample();
            }
            else if(chorus_dry_amount != 1.0)
            {
                samples[c] *= chorus_dry_amount;
            }
            if(waveshaper.mix != 0.0)
            {
                waveshaper.PushSample(samples[c]);
                samples[c] = waveshaper.PopSample();
            }
            if(true)
            {
                limiter.PushSample(samples[c]);
                samples[c] = limiter.PopSample();
            }
        }
    
        public float delay_time = 0.25f;
        public float delay_decay = 0.2f;
        public float delay_stereo_offset = -0.02f;
        public float delay_wet_amount = 0.0f;
        public float delay_dry_amount = 1.0f;
        public int delay_diffusion = 0;
        public float delay_diffusion_ratio = 0.5f;
        
        public Delay delay;// = new Delay(sample_rate, delay_time, delay_decay);
        
        
        public float reverb_dry_amount = 1.0f;
        public float reverb_wet_amount = 0.0f;
        public float reverb_decay = 0.5f;
        public Reverb reverb;// = new Reverb(sample_rate, reverb_decay);
        
        public float lowpass_frequency = 22050.0f;
        public Lowpass lowpass;// = new Lowpass(sample_rate, lowpass_frequency);
        
        public float highpass_frequency = 20.0f;
        public Highpass highpass;// = new Highpass(sample_rate, highpass_frequency);
        
        public float ringmod_frequency = 0.25f;
        public float ringmod_phase = 0.0f;
        public float ringmod_amount = 0.0f;
        public Ringmod ringmod;// = new Ringmod(sample_rate, ringmod_frequency, ringmod_phase, ringmod_amount);
        
        public float limiter_pre_gain = 1.0f;
        public float limiter_lookahead = 0.001f;
        public float limiter_attack = 0.001f;
        public float limiter_sustain = 0.04f;
        public float limiter_release = 0.04f;
        public float limiter_threshold = 1.0f;
        public Limiter limiter;// = new Limiter(sample_rate, limiter_pre_gain, limiter_lookahead, limiter_attack, limiter_sustain, limiter_release, limiter_threshold);
        
        public float flanger_min_depth = 0.002f;
        public float flanger_max_depth = 0.005f;
        public float flanger_cycle_time = 2.0f;
        public float flanger_wet_amount = 0.0f;
        public float flanger_dry_amount = 1.0f;
        public float flanger_feedback = 0.5f;
        
        public Flanger flanger;// = new Flanger(sample_rate, flanger_min_depth, flanger_max_depth, flanger_cycle_time);
        
        public float chorus_depth = 0.002f;
        public float chorus_rate = 5.0f;
        public float chorus_wet_amount = 0.0f;
        public float chorus_dry_amount = 1.0f;
        public int chorus_voices = 3;
        public Chorus chorus;// = new Chorus(sample_rate, chorus_depth, chorus_rate, chorus_wet_amount, chorus_dry_amount, chorus_voices);
        
        public float waveshaper_pre_gain = 1.0f;
        public float waveshaper_exponent = 1.0f;
        public float waveshaper_clip = 1.0f;
        public int waveshaper_clip_mode = 0; // 0: normal clipping. 1: "bounce" clipping. 2: wrapping
        public int waveshaper_quantization = 0; // quantize to number of steps
        public float waveshaper_mix = 0.0f;
        
        public Waveshaper waveshaper;// = new Waveshaper(sample_rate, waveshaper_pre_gain, waveshaper_exponent, waveshaper_clip, waveshaper_clip_mode, waveshaper_quantization, waveshaper_mix);
        
        
        public int gen_nonce = 0;
        public async void Generate()
        {
            gen_nonce += 1;
            var self_nonce = gen_nonce;
            (parent.FindNode("Regen") as Button).Disabled = true;
            (parent.FindNode("Replay") as Button).Disabled = true;
            //pcm_source = MakePcmSource(GD.Load("res://tambourine.wav"));
            //pcm_source = MakePcmSource(GD.Load("res://paper bag.wav"));
            samples = new List<Vector2>();
            Restart();
            
            int aa = oversampling;
            float break_limit = 0.2f;
            int silence_count = 0;
            float silence_limit = 1.0f/32767.0f;
            
            var start_time = OS.GetTicksMsec();
            foreach(var _i in GD.Range((int)(sample_rate*time_limit)))
            {
                if(Mathf.Abs(OS.GetTicksMsec() - start_time) > 10)
                {
                    await ToSignal(parent.GetTree(), "idle_frame");
                    if(gen_nonce != self_nonce)
                    {
                        return;
                    }
                    start_time = OS.GetTicksMsec();
                }
                UpdateEnvelope(gen_time);
                var old_time = gen_time;
                var next = Vector2.Zero;
                var current_freq = freq * SemitonesToFactor(freq_offset_lfo + freq_offset_sweep + freq_offset_step);
                foreach(var _j in GD.Range(aa))
                {
                    gen_cursor += current_freq/sample_rate/aa*2.0f;
                    if(sin_volume != 0.0f)
                    {
                        var f = sin_detune    != 0.0f ? SemitonesToFactor(sin_detune) : 1.0f;
                        next += Vector2.One * sin_volume    * _Sin   (gen_cursor * f);
                    }
                    if(tri_volume != 0.0f)
                    {
                        var f = tri_detune    != 0.0f ? SemitonesToFactor(tri_detune) : 1.0f;
                        next += Vector2.One * tri_volume    * _Tri   (gen_cursor * f, tri_stages);
                    }
                    if(square_volume != 0.0f)
                    {
                        var f = square_detune != 0.0f ? SemitonesToFactor(square_detune) : 1.0f;
                        next += Vector2.One * square_volume * _Square(gen_cursor * f, square_width);
                    }
                    if(saw_volume != 0.0f)
                    {
                        var f = saw_detune    != 0.0f ? SemitonesToFactor(saw_detune) : 1.0f;
                        next += Vector2.One * saw_volume    * _Saw   (gen_cursor * f, saw_exponent);
                    }
                    if(pcm_volume != 0.0f)
                    {
                        next += Vector2.One * pcm_volume    * _Pcm   (gen_cursor);
                    }
                }
                var sample = next * 0.5f / aa * envelope;
                samples.Add(sample);
                gen_time += 1/sample_rate;
                UpdateEvents(old_time);
                UpdateFilters();
                sample = samples[samples.Count-1];
                
                if(Mathf.Abs(sample.x) < silence_limit && Mathf.Abs(sample.y) < silence_limit)
                {
                    silence_count += 1;
                }
                else
                {
                    //print("resetting silence count")
                    silence_count = 0;
                }
                if(silence_count/sample_rate > break_limit)
                {
                    break;
                }
            }
            (parent.FindNode("Regen") as Button).Disabled = false;
            (parent.FindNode("Replay") as Button).Disabled = false;
            EmitSignal("generation_complete");
        
        }
    
        public float step_time = 0.0f;
        public float step_semitones = 4.0f;
        public float step_semitones_stagger = -1.0f;
        public float step_retrigger = 1.0f;
        public float step_loop = 0.0f;
        
        public float freq_lfo_rate = 0.0f;
        public float freq_lfo_strength = 0.0f;
        public float freq_lfo_shape = 0.0f;
        
        public float freq_sweep_rate = 0.0f; // semitones per second
        public float freq_sweep_delta = 0.0f; // semitones per second per second
        
        //# FIXME: AHR volume isn't implemented correctly
        // (note to this: use points 0, 1/2, && 3, !123 || 234)
        public float attack = 0.0f;
        public float attack_exponent = 1.0f;
        public float attack_volume = 1.0f;
        public float hold = 0.2f;
        public float hold_volume = 1.0f;
        public float release = 0.8f;
        public float release_exponent = 4.0f;
        public float release_volume = 1.0f;
        
        public float envelope = 1.0f;
        
        public void UpdateEnvelope(float time)
        {
            if(time < attack && attack > 0.0f)
            {
                envelope = time/attack;
                envelope = Mathf.Pow(envelope, attack_exponent) * attack_volume;
            }
            else if(time - attack < hold)
            {
                envelope = hold_volume;
            }
            else if(time - attack - hold < release)
            {
                if(release > 0.0f)
                {
                    envelope = Mathf.Max(0, 1.0f - ((time - attack - hold)/release));
                    if(release_exponent != 1.0f)
                    {
                        envelope = Mathf.Pow(envelope, release_exponent);
                    }
                    envelope *= release_volume;
                }
                else
                {
                    envelope = 0.0f;
                }
            }
            else
            {
                envelope = 0.0f;
        
            }
        }
    
        public void UpdateEvents(float old_time)
        {
            if(step_time > 0.0f)
            {
                var trigger_time = gen_time;
                var step_semitones = this.step_semitones;
                var step_semitones_stagger = this.step_semitones_stagger;
                if(step_retrigger != 0.0f)
                {
                    if(Mathf.PosMod(old_time, step_time*step_loop) > Mathf.PosMod(trigger_time, step_time*step_loop))
                    {
                        freq_offset_step = 0.0f;
                    }
                    while(step_loop > 0.0f && trigger_time > step_time*step_loop)
                    {
                        trigger_time -= step_time*step_loop;
                        old_time -= step_time*step_loop;
                    }
                    while(old_time > step_time)
                    {
                        old_time -= step_time;
                        trigger_time -= step_time;
                        step_semitones += step_semitones_stagger;
                        step_semitones_stagger = -step_semitones_stagger;
                    }
                }
                if(old_time < step_time && trigger_time >= step_time)
                {
                    freq_offset_step += step_semitones;
            
                }
            }
            if(freq_lfo_strength != 0.0f && freq_lfo_rate != 0.0f)
            {
                var t = gen_time * freq_lfo_rate * 2.0f;
                if(freq_lfo_shape == 0.0f)
                {
                    freq_offset_lfo = _Tri(t);
                }
                else if(freq_lfo_shape == 1.0f)
                {
                    freq_offset_lfo = _Square(t, 0.5f, true);
                }
                else if(freq_lfo_shape == 2.0f)
                {
                    freq_offset_lfo = _Square(t, 0.25f, true);
                }
                else if(freq_lfo_shape == 3.0f)
                {
                    freq_offset_lfo = _Saw(t, 1.0f, true);
                }
                else if(freq_lfo_shape == 4.0f)
                {
                    freq_offset_lfo = _Saw(t, 3.0f, true);
                }
                else if(freq_lfo_shape == 5.0f)
                {
                    if((int)(pcm_source) == 0)
                    {
                        freq_offset_lfo = _Pcm(t / pcm_rate).x;
                    }
                    else
                    {
                        var pcm_data = _Pcm(t / pcm_rate);
                        freq_offset_lfo = (pcm_data.x + pcm_data.y)/2.0f;
                    }
                    if(GD.Randi() % 10000 == 0)
                    {
                        GD.Print(freq_offset_lfo);
                    }
                }
                freq_offset_lfo *= freq_lfo_strength;
            }
            else
            {
                freq_offset_lfo = 0.0f;
            
            }
            var sweep_offset = freq_sweep_delta * gen_time;
            freq_offset_sweep += (freq_sweep_rate + sweep_offset) * 1.0f/sample_rate;
        }
    
        public void Restart()
        {
            freq_offset_lfo = 0.0f; // semitones
            freq_offset_sweep = 0.0f; // semitones
            freq_offset_step = 0.0f; // semitones
            gen_time = 0.0f;
            gen_cursor = 0.0f;
            
            last_sq_cursor = 0.0f;
            last_saw_cursor = 0.0f;
            
            delay = new Delay(sample_rate, delay_time, delay_decay);
            delay.stereo_offset = delay_stereo_offset;
            delay.wet_amount = delay_wet_amount;
            delay.dry_amount = delay_dry_amount;
            delay.diffusion = delay_diffusion;
            delay.diffusion_ratio = delay_diffusion_ratio;
            
            reverb = new Reverb(sample_rate, reverb_decay);
            reverb.wet_amount = reverb_wet_amount;
            reverb.dry_amount = reverb_dry_amount;
            
            lowpass = new Lowpass(sample_rate, lowpass_frequency);
            highpass = new Highpass(sample_rate, highpass_frequency);
            ringmod = new Ringmod(sample_rate, ringmod_frequency, ringmod_phase, ringmod_amount);
            
            flanger = new Flanger(sample_rate, flanger_min_depth, flanger_max_depth, flanger_cycle_time);
            flanger.wet_amount = flanger_wet_amount;
            flanger.dry_amount = flanger_dry_amount;
            flanger.feedback = flanger_feedback;
            
            chorus = new Chorus(sample_rate, chorus_depth, chorus_rate, chorus_wet_amount, chorus_dry_amount, chorus_voices);
            
            waveshaper = new Waveshaper(sample_rate,
                waveshaper_pre_gain,
                waveshaper_exponent,
                waveshaper_clip,
                waveshaper_clip_mode,
                waveshaper_quantization,
                waveshaper_mix
            );
            
            limiter = new Limiter(sample_rate,
                limiter_pre_gain,
                limiter_lookahead,
                limiter_attack,
                limiter_sustain,
                limiter_release,
                limiter_threshold
            );
        
        }
        
        //public signal generation_complete
        [Signal]
        delegate void generation_complete();
    }
    
    public void SetSliderValue(Range slider, object value)
    {
        if (value is int)
        {
            slider.Value = (float)(int)value;
        }
        else if (value is bool)
        {
            slider.Value = ((bool)value) ? 1.0f : 0.0f;
        }
        else
        {
            slider.Value = (float)value;
        }
    }
    
    public void SetValue(String key, object value)
    {  
        SetSliderValue(sliders[key] as Range, value);
    }
    
    public void RandomizeValue(String key, float lower, float upper)
    {  
        Range slider = sliders[key] as Range;
        if(slider.MaxValue <= slider.MinValue)
        {
            return;
        }
        // assigning to .value applies the slider's own limit, then triggers the signal that updates the generator
        if(!slider.ExpEdit || slider.MinValue <= 0.0f)
        {
            slider.Value = GD.RandRange(lower, upper);
        }
        else
        {
            float step = Mathf.Max(0.001f, (float)(slider.Step));
            lower = Mathf.Max(step, lower);
            upper = Mathf.Max(step, upper);
            slider.Value = Mathf.Exp((float)GD.RandRange(Mathf.Log(lower), Mathf.Log(upper)));
        }
    }
    
    public void ResetAllValues()
    {  
        foreach(String key in default_values.Keys)
        {
            if(key == "oversampling" || key == "time_limit")
            {
                continue;
            }
            SetValue(key, default_values[key]);
    
        }
    }
    
    public String RandomChoice(Array array)
    {  
        return (String)array[(int)(GD.Randi() % (long)array.Count)];
    }
    
    public void RandomPickup()
    {  
        GD.Print("seed:");
        GD.Print(OS.GetTicksUsec());
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        SetValue("square_volume", 0.0f);
        
        String which = RandomChoice(new Array(){"square_volume", "tri_volume"});
        SetValue(which, 1.0f);
        
        RandomizeValue("freq", 400.0f, 2400.0f);
        RandomizeValue("hold", 0.0f, 0.1f);
        RandomizeValue("release", 0.1f, 0.5f);
        
        if(GD.Randi() % 2 == 0)
        {
            RandomizeValue("step_time", 0.05f, 0.1f);
            RandomizeValue("step_semitones", 1.0f, 7.0f);
            SetValue("step_retrigger", 0.0f);
        
        }
        generator.Generate();
    }
    
    public void RandomLaser()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        SetValue("square_volume", 0.0f);
        
        RandomizeValue("freq", 400.0f, 4400.0f);
        RandomizeValue("hold", 0.2f, 0.4f);
        RandomizeValue("release", 0.05f, 0.15f);
        
        RandomizeValue("freq_sweep_rate", -48.0f, -256.0f);
        
        var which = RandomChoice(new Array(){"square_volume", "tri_volume", "saw_volume", "sin_volume"});
        SetValue(which, 1.0f);
        if(which == "square_volume")
        {
            RandomizeValue("square_width", 0.5f, 0.8f);
        
        }
        SetValue("highpass_frequency", 100.0f);
        
        generator.Generate();
    }
    
    bool explosion_no_delay = false;
    public void RandomExplosion()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        SetValue("square_volume", 0.0f);
        
        RandomizeValue("freq", 40.0f, 700.0f);
        RandomizeValue("hold", 0.0f, 0.5f);
        RandomizeValue("release", 0.5f, 1.5f);
        
        if(GD.Randi() % 4 > 0)
        {
            RandomizeValue("freq_sweep_rate", -256.0f, 64.0f);
        }
        if(generator.freq > 500.0)
        {
            RandomizeValue("freq_sweep_rate", -256.0f, 0.0f);
        
        }
        if(GD.Randi() % 3 == 0)
        {
            RandomizeValue("freq_lfo_rate", 3.0f, 6.0f);
            RandomizeValue("freq_lfo_strength", 0.5f, 12.0f);
        }
        if(GD.Randi() % 3 == 0)
        {
            SetValue("ringmod_phase", 0.25f);
            RandomizeValue("ringmod_frequency", 1.0f, 4.0f);
            RandomizeValue("ringmod_amount", 0.5f, 1.0f);
        
        }
        if(!explosion_no_delay && GD.Randi() % 3 == 0)
        {
            RandomizeValue("delay_time", 0.2f, 0.4f);
            RandomizeValue("delay_wet_amount", 0.1f, 0.4f);
            RandomizeValue("delay_diffusion", 0.0f, 4.0f);
            SetValue("delay_decay", 0.0f);
            
        }
        SetValue("pcm_volume", 1.0f);
        RandomizeValue("pcm_noise_cycle", 32.0f, 65536.0f);
        
        generator.Generate();
    }
    
    public void RandomPowerup()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        SetValue("square_volume", 0.0f);
        
        var which = RandomChoice(new Array(){"square_volume", "tri_volume"});
        SetValue(which, 1.0f);
        
        RandomizeValue("freq", 200.0f, 2400.0f);
        RandomizeValue("attack", 0.0f, 0.02f);
        RandomizeValue("hold", 0.3f, 0.5f);
        RandomizeValue("release", 0.5f, 1.0f);
        
        var slide_type = GD.Randi() % 3;
        if(slide_type == 0)
        {
            RandomizeValue("step_time", 0.02f, 0.05f);
            RandomizeValue("step_semitones", 1.0f, 7.0f);
            SetValue("step_semitones_stagger", 0.0f);
            SetValue("step_retrigger", 1.0f);
            RandomizeValue("step_loop", 3.0f, 6.0f);
        }
        else if(slide_type == 1)
        {
            RandomizeValue("freq_sweep_rate", 7.0f, 128.0f);
        }
        else
        {
            RandomizeValue("freq_lfo_rate", 3.0f, 20.0f);
            RandomizeValue("freq_lfo_strength", 6.0f, 24.0f);
            SetValue("freq_lfo_shape", 3.0f);
        }
        if(slide_type != 2 && GD.Randi() % 2 == 0)
        {
            RandomizeValue("freq_lfo_rate", 3.0f, 6.0f);
            RandomizeValue("freq_lfo_strength", 0.2f, 2.0f);
        }
        if(GD.Randi() % 4 == 0)
        {
            SetValue("delay_wet_amount", 0.2f);
        }
        if(GD.Randi() % 4 == 0)
        {
            SetValue("chorus_wet_amount", 0.2f);
        }
        
        generator.Generate();
    }
    
    public void RandomJump()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        
        RandomizeValue("square_width", 0.5f, 0.8f);
        
        RandomizeValue("freq", 50.0f, 400.0f);
        RandomizeValue("hold", 0.1f, 0.5f);
        RandomizeValue("release", 0.1f, 0.2f);
        
        RandomizeValue("freq_sweep_rate", 15.0f, 128.0f);
        
        generator.Generate();
    }
    
    public void RandomHit()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        if(GD.Randi() % 4 == 0)
        {
            explosion_no_delay = true;
            RandomExplosion();
            explosion_no_delay = false;
        }
        else
        {
            RandomLaser();
        }
        RandomizeValue("freq", 200.0f, 1600.0f);
        RandomizeValue("hold", 0.05f, 0.10f);
        RandomizeValue("release", 0.15f, 0.25f);
        RandomizeValue("freq_sweep_rate", -128.0f, -512.0f);
        
        generator.Generate();
    }
    
    public void RandomBlip()
    {  
        GD.Seed(OS.GetTicksUsec() ^ 1234895143);
        ResetAllValues();
        SetValue("square_volume", 0.0f);
        
        var which = RandomChoice(new Array(){"square_volume", "tri_volume", "sin_volume"});
        SetValue(which, 1.0f);
        
        RandomizeValue("freq", 200.0f, 2400.0f);
        RandomizeValue("hold", 0.1f, 0.2f);
        RandomizeValue("release", 0.01f, 0.05f);
        
        if(GD.Randi() % 2 == 0)
        {
            RandomizeValue("step_time", 0.02f, 0.085f);
            RandomizeValue("step_semitones", 24.0f, -24.0f);
            SetValue("step_retrigger", 0.0f);
        
        }
        
        generator.Generate();
    }
    
    Node control_target;
    
    public void SetLabelValue(Label label, float value)
    {  
        if(Mathf.Abs(value) == 0.0f)
        {
            label.Text = "0.00";
        }
        else if(Mathf.Abs(value) < 0.1)
        {
            label.Text = $"{value:F3}";
        }
        else if(Mathf.Abs(value) < 10)
        {
            label.Text = $"{value:F2}";
            //label.Text = "%.2f" % value;
        }
        else if(Mathf.Abs(value) < 1000)
        {
            label.Text = $"{value:F1}";
            //label.Text = "%.1f" % value;
        }
        else
        {
            label.Text = $"{value:0}";
            //label.Text = "%.0f" % value;
        }
    }
    
    public void SliderUpdate(float value, Range _slider, Label number, String name)
    {  
        if (generator.Get(name) is int)
        {
            SetLabelValue(number, (float)value);
            generator.Set(name, (int)value);
        }
        else
        {
            SetLabelValue(number, value);
            generator.Set(name, (float)value);
        }
        //GD.Print(name);
        //GD.Print(generator.Get(name));
    }
    
    public Godot.Collections.Dictionary<String, object> default_values
        = new Godot.Collections.Dictionary<String, object>(){};
    public Godot.Collections.Dictionary<String, Range> sliders
        = new Godot.Collections.Dictionary<String, Range>(){};
    
    public Slider AddSlider(String name, float MinValue, float MaxValue)
    {  
        var value = generator.Get(name);
        
        var label = new Label();
        label.Text = name.Capitalize();
        
        var number = new Label();
        if (value is int)
        {
            SetLabelValue(number, (float)(int)value);
        }
        else if (value is bool)
        {
            SetLabelValue(number, ((bool)value) ? 1.0f : 0.0f);
        }
        else
        {
            SetLabelValue(number, (float)value);
        }
        number.SizeFlagsHorizontal |= (int)SizeFlags.ExpandFill;
        number.SizeFlagsStretchRatio = 0.25f;
        number.Align = Label.AlignEnum.Right;
        number.ClipText = true;
        
        var slider = new HSlider();
        slider.Name = name;
        slider.MinValue = MinValue;
        slider.MaxValue = MaxValue;
        slider.Step = 0.01f;
        slider.SizeFlagsHorizontal |= (int)SizeFlags.ExpandFill;
        slider.SizeFlagsStretchRatio = 0.75f;
        slider.TickCount = 5;
        slider.TicksOnBorders = true;
        slider.AddToGroup("Slider");
        sliders[name] = slider;
        
        slider.Connect("value_changed", this, "SliderUpdate", new Godot.Collections.Array(){slider, number, name});
        SetSliderValue(slider, value);
        
        var container = new HSplitContainer();
        container.DraggerVisibility = SplitContainer.DraggerVisibilityEnum.Hidden;
        container.AddChild(number);
        container.AddChild(slider);
        
        control_target.AddChild(label);
        control_target.AddChild(container);
        return slider;
    }
    
    public void AddSeparator()
    {  
        var separator = new HSeparator();
        control_target.AddChild(separator);
    }
    
    public async void AddControls()
    {  
        control_target = GetNode("VBox/Scroll/Box/A/Scroller/Controls");
        
        Slider slider;
        slider = AddSlider("freq", 20.0f, 22050.0f);
        slider.ExpEdit = true;
        slider.Step = 0.5f;
        AddSeparator();
        AddSlider("square_volume", -1.0f, 1.0f);
        AddSlider("square_detune", -48.0f, 48.0f);
        AddSlider("square_width", 0.0f, 1.0f);
        AddSeparator();
        AddSlider("tri_volume", -1.0f, 1.0f);
        AddSlider("tri_detune", -48.0f, 48.0f);
        slider = AddSlider("tri_stages", 0.0f, 32.0f);
        slider.Step = 1.0f;
        AddSeparator();
        AddSlider("saw_volume", -1.0f, 1.0f);
        AddSlider("saw_detune", -48.0f, 48.0f);
        slider = AddSlider("saw_exponent", 0.01f, 16.0f);
        slider.ExpEdit = true;
        AddSeparator();
        AddSlider("sin_volume", -1.0f, 1.0f);
        AddSlider("sin_detune", -48.0f, 48.0f);
        AddSeparator();
        AddSlider("pcm_volume", -1.0f, 1.0f);
        AddSlider("pcm_offset", 0.0f, 5.0f);
        AddSlider("pcm_cutoff", 0.0f, 15.0f);
        AddSlider("pcm_rate", 0.01f, 100.0f).ExpEdit = true;
        slider = AddSlider("pcm_noise_cycle", 2.0f, Mathf.Pow(2.0f, 16.0f));
        slider.ExpEdit = true;
        slider.Step = 1.0f;
        AddSlider("pcm_source", 0.0f, 5.0f).Step = 1;
        AddSlider("pcm_sample_loop", 0.0f, 1.0f).Step = 1;
        
        control_target = GetNode("VBox/Scroll/Box/B/Scroller/Controls");
        
        AddSlider("step_time", 0.0f, 5.0f);
        AddSlider("step_semitones", -48.0f, 48.0f);
        AddSlider("step_semitones_stagger", -48.0f, 48.0f);
        AddSlider("step_retrigger", 0.0f, 1.0f).Step = 1.0f;
        AddSlider("step_loop", 0.0f, 16.0f).Step = 1.0f;
        AddSeparator();
        AddSlider("freq_lfo_rate", 0.0f, 50.0f);
        AddSlider("freq_lfo_strength", -12.0f, 12.0f);
        AddSlider("freq_lfo_shape", 0.0f, 5.0f).Step = 1.0f;
        AddSeparator();
        AddSlider("freq_sweep_rate", -12.0f*32.0f, 12.0f*32.0f).Step = 1.0f;
        AddSlider("freq_sweep_delta", -12.0f*32.0f, 12.0f*32.0f).Step = 1.0f;
        AddSeparator();
        AddSlider("ringmod_frequency", 0.01f, 22050.0f).ExpEdit = true;
        AddSlider("ringmod_phase", 0.0f, 1.0f);
        AddSlider("ringmod_amount", -2.0f, 2.0f);
        
        control_target = GetNode("VBox/Scroll/Box/C/Scroller/Controls");
        
        slider = AddSlider("time_limit", 0.1f, 50.0f);
        slider.ExpEdit = true;
        AddSeparator();
        AddSlider("attack", 0.0f, 5.0f);
        slider = AddSlider("attack_exponent", 0.1f, 10.0f);
        slider.ExpEdit = true;
        AddSlider("attack_volume", -1.0f, 1.0f);
        AddSeparator();
        AddSlider("hold", 0.0f, 10.0f);
        AddSlider("hold_volume", -1.0f, 1.0f);
        AddSeparator();
        AddSlider("release", 0.01f, 20.0f).ExpEdit = true;
        slider = AddSlider("release_exponent", 0.1f, 10.0f);
        slider.ExpEdit = true;
        AddSlider("release_volume", -1.0f, 1.0f);
        AddSeparator();
        AddSlider("limiter_pre_gain", 0.05f, 20.0f);
        AddSlider("limiter_lookahead", 0.0f, 0.01f).Step = 0.001f;
        AddSlider("limiter_attack", 0.0f, 0.01f).Step = 0.001f;
        AddSlider("limiter_sustain", 0.0f, 0.5f);
        AddSlider("limiter_release", 0.0f, 0.5f);
        AddSlider("limiter_threshold", 0.0f, 1.0f);
        
        control_target = GetNode("VBox/Scroll/Box/D/Scroller/Controls");
        
        AddSlider("oversampling", 1.0f, 8.0f).Step = 1.0f;
        AddSeparator();
        AddSlider("delay_time", 0.001f, 4.0f).Step = 0.001f;
        AddSlider("delay_decay", 0.0f, 2.0f);
        AddSlider("delay_stereo_offset", -1.0f, 1.0f).Step = 0.001f;
        AddSlider("delay_wet_amount", -1.0f, 1.0f);
        AddSlider("delay_dry_amount", -1.0f, 1.0f);
        AddSlider("delay_diffusion", 0.0f, 8.0f).Step = 1.0f;
        AddSlider("delay_diffusion_ratio", 0.0f, 1.0f);
        AddSeparator();
        AddSlider("reverb_decay", 0.0f, 0.9f);
        AddSlider("reverb_wet_amount", -8.0f, 8.0f);
        AddSlider("reverb_dry_amount", -1.0f, 1.0f);
        AddSeparator();
        AddSlider("lowpass_frequency", 20.0f, 22050.0f).ExpEdit = true;
        AddSeparator();
        AddSlider("highpass_frequency", 20.0f, 22050.0f).ExpEdit = true;
        AddSeparator();
        AddSlider("flanger_min_depth", 0.0f, 0.5f);
        AddSlider("flanger_max_depth", 0.0f, 0.5f);
        AddSlider("flanger_cycle_time", 0.01f, 20.0f).ExpEdit = true;
        AddSlider("flanger_wet_amount", -1.0f, 1.0f);
        AddSlider("flanger_dry_amount", -1.0f, 1.0f);
        AddSlider("flanger_feedback", 0.0f, 1.0f);
        AddSeparator();
        AddSlider("chorus_depth", 0.0f, 0.05f).Step = 0.001f;
        AddSlider("chorus_rate", 0.0f, 50.0f);
        AddSlider("chorus_wet_amount", 0.0f, 1.0f);
        AddSlider("chorus_dry_amount", 0.0f, 1.0f);
        AddSlider("chorus_voices", 0.0f, 8.0f).Step = 1.0f;
        AddSeparator();
        AddSlider("waveshaper_pre_gain", 0.0f, 16.0f);
        AddSlider("waveshaper_exponent", 0.1f, 10.0f).ExpEdit = true;
        AddSlider("waveshaper_clip", 0.0f, 8.0f);
        AddSlider("waveshaper_clip_mode", 0.0f, 2.0f).Step = 1.0f;
        AddSlider("waveshaper_quantization", 0.0f, 256.0f).Step = 1.0f;
        AddSlider("waveshaper_mix", 0.0f, 1.0f);
        
        await ToSignal(GetTree(), "idle_frame");
        foreach(var _slider in GetTree().GetNodesInGroup("Slider"))
        {
            var slider2 = _slider as Slider;
            GD.Print(slider2.Name);
            var value = generator.Get(slider2.Name);
            default_values[slider2.Name] = value;
            SetSliderValue(slider2, value);
        }
    }
    
    public void _OnFilesDropped(String[] files, int _screen)
    {
        var AudioLoader = (GDScript) GD.Load("res://GDScriptAudioImport.gd");
        var audio_loader = (Godot.Object) AudioLoader.New();
        var stream = audio_loader.Call("loadfile", files[0]);
        if(!(stream is AudioStreamSample))
        {
            return;
        }
        generator.pcm_source_custom = Generator.MakePcmSource(stream as AudioStreamSample);
        generator.pcm_source = generator.pcm_sources.Count+1;
        var player = GetNode("VBox/Scroll/Box/A/Scroller/Controls").FindNode("pcm_source", true, false) as Slider;
        player.Value = generator.pcm_source;
    }
    
    public String fname;
    public String fname_bare;
    public void Save()
    {  
        var dir = new Directory();
        dir.MakeDir("sfx_output");
        
        var timestamp = OS.GetSystemTimeMsecs();
        fname = $"sfx_output/sfx_{timestamp}.wav";
        fname_bare = $"sfx_{timestamp}.wav";
        var bytes = new StreamPeerBuffer();
        foreach(Vector2 vec in generator.samples)
        {
            var x = Mathf.Clamp(vec.x, -1.0f, 1.0f);
            var y = Mathf.Clamp(vec.y, -1.0f, 1.0f);
            bytes.Put16((short)(x*32767.0f));
            bytes.Put16((short)(y*32767.0f));
        }
        bytes.Seek(0);
        var audio = new AudioStreamSample();
        audio.Format = AudioStreamSample.FormatEnum.Format16Bits;
        audio.Stereo = true;
        GD.Print(bytes.GetAvailableBytes());
        audio.Data = bytes.DataArray;
        
        if(OS.GetName() == "HTML5")
        {
            fname = "user://" + fname_bare;
        
        }
        audio.SaveToWav(fname);
        var file = new File();
        file.Open(fname, File.ModeFlags.Read);
        
        JavaScript.DownloadBuffer(file.GetBuffer((long)file.GetLen()), fname_bare, "audio/wave");
    }
    
    public async void UpdatePlayer()
    {  
        var player = GetNode("Player") as AudioStreamPlayer;
        player.Stop();
        
        await ToSignal(GetTree(), "idle_frame");
        await ToSignal(GetTree(), "idle_frame");
        
        var bytes = new StreamPeerBuffer();
        foreach(var vec in generator.samples)
        {
            var x = Mathf.Clamp(vec.x, -1.0f, 1.0f);
            var y = Mathf.Clamp(vec.y, -1.0f, 1.0f);
            bytes.Put16((short)(x*32767.0f));
            bytes.Put16((short)(y*32767.0f));
        }
        bytes.Seek(0);
        var audio = new AudioStreamSample();
        audio.Format = AudioStreamSample.FormatEnum.Format16Bits;
        audio.Stereo = true;
        audio.Data = bytes.DataArray;
        player.Stream = audio;
        
        await ToSignal(GetTree(), "idle_frame");
        await ToSignal(GetTree(), "idle_frame");
        await ToSignal(GetTree(), "idle_frame");
        
        player.Play();
    }
    
    public void Replay()
    {
        var player = GetNode("Player") as AudioStreamPlayer;
        player.Stop();
        player.Play();
    }
    
    public Generator generator;
    
    public override async void _Ready()
    {
        GD.Print("initializing");
        
        var scroll_area = GetNode("VBox/Scroll") as ScrollContainer;
        var scrollbar = scroll_area.GetHScrollbar();
        scrollbar.AddIconOverride("decrement", GD.Load("res://left.png") as Texture);
        scrollbar.AddIconOverride("decrement_highlight", GD.Load("res://lefthover.png") as Texture);
        scrollbar.AddIconOverride("decrement_pressed", GD.Load("res://leftclick.png") as Texture);
        scrollbar.AddIconOverride("increment", GD.Load("res://right.png") as Texture);
        scrollbar.AddIconOverride("increment_highlight", GD.Load("res://righthover.png") as Texture);
        scrollbar.AddIconOverride("increment_pressed", GD.Load("res://rightclick.png") as Texture);
        scrollbar.CustomStep = 128;
        
        var _unused = GetTree().Connect("files_dropped", this, "_OnFilesDropped");
        
        generator = new Generator();
        generator.parent = this;
        AddControls();
        
        await ToSignal(GetTree(), "idle_frame");
        await ToSignal(GetTree(), "idle_frame");
        await ToSignal(GetTree(), "idle_frame");
        
        _unused = generator.Connect("generation_complete", this, "UpdatePlayer");
        generator.Generate();
        
        _unused = GetNode("VBox/Buttons/Regen").Connect("pressed", generator, "Generate");
        _unused = GetNode("VBox/Buttons/Replay").Connect("pressed", this, "Replay");
        _unused = GetNode("VBox/Buttons/Save").Connect("pressed", this, "Save");
        
        _unused = GetNode("VBox/Buttons2/Pickup").Connect("pressed", this, "RandomPickup");
        _unused = GetNode("VBox/Buttons2/Laser").Connect("pressed", this, "RandomLaser");
        _unused = GetNode("VBox/Buttons2/Explosion").Connect("pressed", this, "RandomExplosion");
        _unused = GetNode("VBox/Buttons2/Powerup").Connect("pressed", this, "RandomPowerup");
        _unused = GetNode("VBox/Buttons2/Hit").Connect("pressed", this, "RandomHit");
        _unused = GetNode("VBox/Buttons2/Jump").Connect("pressed", this, "RandomJump");
        _unused = GetNode("VBox/Buttons2/Blip").Connect("pressed", this, "RandomBlip");
    }
}
