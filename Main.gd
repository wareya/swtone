extends Control

class Filter extends Reference:
    var sample_rate = 1.0
    func _init(sample_rate):
        self.sample_rate = sample_rate
    func push_sample(_x):
        pass
    func pop_sample():
        pass

class Delay extends Filter:
    var wet_amount = 0.25
    var dry_amount = 1.0
    var decay = 0.2
    var stereo_offset = -0.01
    var diffusion = 8
    var diffusion_ratio = 1.0
    
    var wet = []
    var dry = Vector2.ZERO
    var cursor = 0
    func _init(sample_rate, time, decay).(sample_rate):
        self.decay = decay
        wet = []
        for _i in range(time*sample_rate):
            wet.push_back(Vector2.ZERO)
    func push_sample(x):
        wet[cursor] *= decay
        wet[cursor] += x
        dry = x
        cursor = (cursor + 1) % wet.size()
    
    func sample(c):
        return wet[(int(c) % wet.size() + wet.size() ) % wet.size()]
    
    func pull_from_cursor(c):
        var _wet = sample(c)
        var offset = floor(abs(stereo_offset * wet.size()))
        if diffusion > 0:
            offset = floor(offset / (diffusion + 1.0))
        if stereo_offset > 0.0:
            _wet.y = sample(c+offset).y
        elif stereo_offset < 0.0:
            _wet.x = sample(c+offset).x
        return _wet
        
    func pop_sample():
        if diffusion == 0:
            return dry * dry_amount + pull_from_cursor(cursor) * wet_amount
        else:
            var _wet = Vector2()
            for i in range(diffusion+1):
                var d = float(i)/(diffusion+1)
                seed(i ^ hash(wet.size() ^ 588123))
                d += rand_range(-0.5, 0.5)/(diffusion+1)
                d *= diffusion_ratio
                d = int(d*wet.size())
                _wet += pull_from_cursor(cursor + d)
            _wet /= diffusion+1
            return dry*dry_amount + _wet*wet_amount

class Reverb extends Filter:
    var dry_amount = 1.0
    var wet_amount = 1.0
    var decay = 0.5
    
    var delay1 : Delay
    var delay2 : Delay
    var delay3 : Delay
    var lowpass : Lowpass
    var dry = Vector2()
    func _init(sample_rate, _decay).(sample_rate):
        var decay = _decay
        delay1 = Delay.new(sample_rate, 0.19, decay)
        delay1.wet_amount = 1.0
        delay1.dry_amount = 1.0/8.0
        delay1.diffusion = 8
        delay1.diffusion_ratio = 1.0
        
        delay2 = Delay.new(sample_rate, 0.14, 0.0)
        delay2.wet_amount = 1.0
        delay2.dry_amount = 1.0/8.0
        delay2.diffusion = 8
        delay2.stereo_offset = 0.02
        delay2.diffusion_ratio = 1.0
        
        delay3 = Delay.new(sample_rate, 0.031, 0.0)
        delay3.wet_amount = 1.0
        delay3.dry_amount = 1.0/8.0
        delay3.diffusion = 8
        delay2.stereo_offset = 0.5
        delay3.diffusion_ratio = 1.0
        
        lowpass = Lowpass.new(sample_rate, 2000.0)
        
    func push_sample(x):
        dry = x
    
    func pop_sample():
        delay1.push_sample(dry)
        delay2.push_sample(delay1.pop_sample())
        delay3.push_sample(delay2.pop_sample())
        lowpass.push_sample(delay3.pop_sample())
        return dry * dry_amount + lowpass.pop_sample() * wet_amount * 8.0

class Lowpass extends Filter:
    var cutoff = 0.0
    var decay_constant = 0.0
    var memory = Vector2()
    func _init(sample_rate, cutoff : float).(sample_rate):
        update_decay_constant(cutoff)
    func update_decay_constant(cutoff : float):
        self.cutoff = cutoff
        var y = 1 - cos(cutoff / (sample_rate/2.0) * PI)
        decay_constant = -y + sqrt(y*y + 2*y)
    func push_sample(x):
        memory = memory.linear_interpolate(x, decay_constant)
    func pop_sample():
        return memory

class Highpass extends Filter:
    var cutoff = 0.0
    var decay_constant = 0.0
    var memory = Vector2()
    var memory2 = Vector2()
    func _init(sample_rate, cutoff : float).(sample_rate):
        update_decay_constant(cutoff)
    func update_decay_constant(cutoff : float):
        self.cutoff = cutoff
        var y = 1 - cos(cutoff / (sample_rate/2.0) * PI)
        decay_constant = -y + sqrt(y*y + 2*y)
    func push_sample(x):
        memory = memory.linear_interpolate(x, decay_constant)
        memory2 = x
    func pop_sample():
        return memory2 - memory

class Ringmod extends Filter:
    var frequency : float = 0.0
    var amount : float = 1.0
    var cursor : float = 0.0
    var phase : float = 0.0
    var memory = Vector2()
    func _init(sample_rate, frequency : float, phase : float, amount : float).(sample_rate):
        self.frequency = frequency
        self.amount = amount
        self.phase = phase
        cursor = 0.0
    func push_sample(x):
        memory = x
    func pop_sample():
        var r = memory * sin(cursor*PI*2 + phase*PI*2.0)
        cursor += frequency*2.0/sample_rate
        return memory.linear_interpolate(r, amount)

class Flanger extends Filter:
    var min_depth : int = 0
    var max_depth : int = 0
    var rate = 2.0
    var wet_amount = 1.0
    var dry_amount = 1.0
    var feedback = 0.5
    
    var cursor : int = 0
    var wet = []
    #var stereo_offset = -0.01
    func _init(sample_rate, _min_depth, _max_depth, _rate).(sample_rate):
        self.min_depth = int(_min_depth * sample_rate)
        self.max_depth = int(_max_depth * sample_rate)
        self.rate = float(_rate)
        wet = []
        var count = max(min_depth, max_depth)
        for _i in range(count):
            wet.push_back(Vector2.ZERO)
        if wet.size() == 0:
            wet.push_back(Vector2.ZERO)
    
    func push_sample(x):
        #wet[cursor] *= decay
        #wet[cursor] += x
        wet[cursor % wet.size()] = x
        cursor += 1
    
    func _tri(x):
        return abs((fmod((x*2.0)+1.0,2.0)-1))
    
    func pop_sample():
        var lfo_amount = _tri(cursor/sample_rate * ((1.0/rate) if rate > 0.0 else 0.0))
        var samples_into_past = lerp(min_depth, max_depth, lfo_amount)
        samples_into_past = int(min(samples_into_past, wet.size()-1))
        #if randi() % 999 == 0:
        #    print("%s/%s" % [samples_into_past, wet.size()])
        var _wet = wet[(cursor - 1 - samples_into_past + wet.size()*2) % wet.size()]
        var c = (cursor - 1) % wet.size()
        var dry =  wet[c]
        var out = dry * dry_amount + _wet * wet_amount
        wet[c] *= 1.0 - feedback
        wet[c] += out * feedback
        return out

# TODO add stereo thing somehow
class Chorus extends Filter:
    #var min_depth : int = 0
    var depth : int = 0
    var rate = 2.0
    var wet_amount = 1.0
    var dry_amount = 0.0
    var voices = 3
    
    var cursor : int = 0
    var wet = []
    #var stereo_offset = -0.01
    func _init(sample_rate, depth, _rate, _wet, _dry, _voices).(sample_rate):
        rate = float(_rate)
        wet_amount = _wet
        dry_amount = _dry
        voices = _voices
        
        wet = []
        for _i in range(depth*sample_rate):
            wet.push_back(Vector2.ZERO)
        if wet.size() == 0:
            wet.push_back(Vector2.ZERO)
    
    func push_sample(x):
        wet[cursor % wet.size()] = x
        cursor += 1
    
    func _tri(x):
        return abs((fmod((x*2.0)+1.0,2.0)-1))
    
    func pop_sample():
        var lfo_sum = Vector2()
        for i in range(voices):
            var lfo_amount = -cos((cursor*rate/sample_rate + 2.0*i/voices) * PI)/2.0 + 0.5
            var samples_into_past = int((wet.size()-1) * lfo_amount)
            #if randi() % 999 == 0 and i == 0:
            #    print("%s/%s" % [samples_into_past, wet.size()])
            var _wet = wet[(cursor - 1 - samples_into_past + wet.size()*2) % wet.size()]
            lfo_sum += _wet
        var c = (cursor - 1) % wet.size()
        var dry =  wet[c]    
        var out = dry * dry_amount + lfo_sum * wet_amount
        return out

# TODO: add optional reference low pass
class Waveshaper extends Filter:
    var pre_gain = 1.0
    var exponent = 1.0
    var clip = 1.0
    var clip_mode = 0 # 0: normal clipping. 1: "bounce" clipping. 2: wrapping
    var quantization = 0 # quantize to number of steps
    var mix = 0.0
    
    var memory = Vector2()
    func _init(sample_rate, pre_gain, exponent, clip, clip_mode, quantization, mix).(sample_rate):
        self.pre_gain = pre_gain
        self.exponent = exponent
        self.clip = clip
        self.clip_mode = clip_mode
        self.quantization = quantization
        self.mix = mix
        pass
    func push_sample(x):
        memory = x
        
    func _saw(x):
        var r = fmod(fmod(x+1.0, 2.0) + 2.0, 2.0) - 1.0
        return r
    
    func _tri(x):
        return abs(fmod(fmod(x-1.0, 4.0) + 4.0, 4.0) - 2.0) - 1.0
    func pop_sample():
        var shaped : Vector2 = memory*pre_gain
        shaped = shaped.abs()
        shaped.x = pow(shaped.x, exponent)
        shaped.y = pow(shaped.y, exponent)
        shaped = memory.sign() * shaped
        if clip > 0.0:
            if clip_mode == 0:
                shaped.x = clamp(shaped.x, -clip, clip)
                shaped.y = clamp(shaped.y, -clip, clip)
            elif clip_mode == 1:
                shaped.x = _tri(shaped.x/clip) * clip
                shaped.y = _tri(shaped.y/clip) * clip
            elif clip_mode == 2:
                shaped.x = _saw(shaped.x/clip) * clip
                shaped.y = _saw(shaped.y/clip) * clip
        else:
            shaped = Vector2()
            
        var stages = quantization/2.0 - 0.5
        if stages > 0.0:
            shaped.x = floor(shaped.x*stages)/stages + 1.0/stages/2.0
            shaped.y = floor(shaped.y*stages)/stages + 1.0/stages/2.0
        
        return lerp(memory, shaped, mix)

class Limiter extends Filter:
    var pre_gain = 1.0
    var lookahead = 0.001
    var attack = 0.001
    var sustain = 0.01
    var release = 0.01
    var threshold = 1.0
    
    var amplitude = 1.0
    var hit_time = 0.0
    var hit_amplitude = 1.0
    var time = 0.0
    
    var buffer = []
    var buffer_max = []
    var buffer_max_bucket = []
    var buffer_max_bucket_dirty = []
    var bucket_size = 128
    var cursor = 0
    
    func _init(sample_rate, _pre_gain, _lookahead, _attack, _sustain, _release, _threshold).(sample_rate):
        pre_gain = _pre_gain
        lookahead = _lookahead
        attack = _attack
        sustain = _sustain
        release = _release
        threshold = _threshold
        for _i in range(lookahead * sample_rate + 1):
            buffer.push_back(Vector2())
        for i in range((sustain + lookahead) * sample_rate):
            buffer_max.push_back(0.0)
            if int(i) % bucket_size == 0:
                buffer_max_bucket.push_back(0.0)
                buffer_max_bucket_dirty.push_back(false)
        if buffer.size() == 0:
            buffer.push_back(Vector2())
        if buffer_max.size() == 0:
            buffer_max.push_back(0.0)
            buffer_max_bucket.push_back(0.0)
            buffer_max_bucket_dirty.push_back(false)
        time = 0.0
    
    func max_buffer():
        for i in range(buffer_max_bucket_dirty.size()):
            if buffer_max_bucket_dirty[i]:
                var y = 0.0
                for j in range(i*bucket_size, min((i+1)*bucket_size, buffer_max.size())):
                    var n = buffer_max[j]
                    y = max(n, y)
                buffer_max_bucket[i] = y
                buffer_max_bucket_dirty[i] = false
        var x = 0.0
        for n in buffer_max_bucket:
            x = max(n, x)
        return x
    
    var former_amplitude = 1.0
    func push_sample(x):
        if threshold == 0.0:
            return
        cursor += 1
        x *= pre_gain
        time += 1.0/sample_rate
        
        buffer[cursor % buffer.size()] = x
        var loudness = max(abs(x.x), abs(x.y))
        buffer_max[cursor % buffer_max.size()] = loudness
        buffer_max_bucket_dirty[floor((cursor % buffer_max.size())/bucket_size)] = true
        
        var envelope = max_buffer()
        
        if release > 0:
            var time_since_hit = time - hit_time
            if time_since_hit < release:
                var decay_progress = time_since_hit/release
                amplitude = lerp(hit_amplitude, 1.0, decay_progress)
            else:
                amplitude = 1.0
                
            var amplitude_follower = threshold/max(envelope, threshold)
            if amplitude_follower <= amplitude:
                hit_time = time
                hit_amplitude = amplitude_follower
                amplitude = hit_amplitude
        else:
            amplitude = threshold/max(envelope, threshold)
        handle_attack()
    
    # this "attack" algorithm is numerically unstable but very fast
    # and the instability doesn't matter for the mere seconds/minutes of audio swtone produces
    var attack_amplitude_raw = 1.0
    var attack_amplitude = 1.0
    var attack_buffer = []
    var attack_cursor = 0
    func handle_attack():
        if attack == 0.0:
            attack_amplitude = amplitude
            return
        if attack_buffer.size() == 0:
            attack_amplitude = 0.0
            for _i in range(int(attack * sample_rate + 1)):
                attack_buffer.push_back(amplitude)
                attack_amplitude += amplitude
            attack_amplitude /= attack_buffer.size()
        
        var old = attack_buffer[attack_cursor % attack_buffer.size()]
        attack_buffer[attack_cursor % attack_buffer.size()] = amplitude
        attack_cursor += 1
        
        attack_amplitude = 0.0
        for b in attack_buffer:
            attack_amplitude += b / attack_buffer.size()
        
        #attack_amplitude -= old/attack_buffer.size()
        #attack_amplitude += amplitude/attack_buffer.size()
    
    func pop_sample():
        if threshold == 0.0:
            return Vector2()
        if lookahead > 0.0:
            return buffer[(cursor + 1) % buffer.size()] * attack_amplitude
        else:
            return buffer[cursor % buffer.size()] * attack_amplitude

class Generator extends Reference:
    var parent : Node
    var samples = PoolVector2Array()
    var playback_cursor = 0
    var sample_rate = 44100.0
    var freq = 440.0
    var freq_offset_lfo = 0.0 # semitones
    var freq_offset_sweep = 0.0 # semitones
    var freq_offset_step = 0.0 # semitones
    
    func semitones_to_factor(x):
        return pow(2, x/12.0)
    func factor_to_semitones(x):
        return log(x)/log(2)*12.0
    
    var oversampling = 1.0
    
    var gen_time = 0.0
    var gen_cursor = 0.0
    func _sin(cursor):
        return sin(cursor*PI)
    
    func _tri(cursor, stages = 0.0):
        stages = stages/2.0 - 0.5
        var x = cursor
        x = abs((fmod(x+1.5,2.0)-1))*2-1
        if stages <= 0.0:
            return x
        else:
            return floor(x*stages)/stages + 1.0/stages/2.0
    
    var last_sq_cursor = gen_cursor
    func _square(cursor, width = 0.5, nooffset = false):
        var x = fmod(cursor, 2.0)/2.0
        var out = -1.0 if x < width else 1.0
        
        # FIXME make this work even when the cursor is cycling "backwards"
        if last_sq_cursor < width and x >= width:
            var between = inverse_lerp(x, last_sq_cursor, width)
            out = lerp(-1.0, 1.0, between)
        elif x < last_sq_cursor:
            var between = inverse_lerp(x + 1.0, last_sq_cursor, 1.0)
            out = lerp(1.0, -1.0, between)
        
        var dc_bias = (width - 0.5) * 2
        if nooffset:
            dc_bias = 0.0
        last_sq_cursor = x
        return out + dc_bias
    
    var last_saw_cursor = 0.0
    func _saw(cursor, exponent = 1.0):
        var n = fmod(cursor, 2.0)-1.0
        var out = pow(abs(n), exponent)*sign(n)
        # FIXME make work properly when cursor is moving "backwards"
        if n < last_saw_cursor and (last_saw_cursor - n) >= 1.0:
            out = lerp(-1.0, 1.0, inverse_lerp(last_saw_cursor, n + 2.0, 1.0))
        last_saw_cursor = n
        return out
    
    func make_pcm_source(clip : AudioStreamSample):
        var bytes_per_sample = 1.0
        if clip.format == clip.FORMAT_16_BITS:
            bytes_per_sample *= 2
        if clip.stereo:
            bytes_per_sample *= 2
        var sample_count = clip.data.size() / bytes_per_sample
        print(sample_count)
        var stream = StreamPeerBuffer.new()
        print("error?")
        stream.put_data(clip.data)
        stream.seek(0)
        var samples = []
        if clip.format == clip.FORMAT_16_BITS:
            for _i in sample_count:
                var l = stream.get_16()/32768.0
                var r = stream.get_16()/32768.0 if clip.stereo else l
                var sample = Vector2(l, r)
                if randi() % 10000 == 0:
                    print(sample)
                samples.push_back(sample)
            return samples
    
    #var pcm_source = null
    #var pcm_source = null
    var pcm_sources = [
        make_pcm_source(preload("res://paper bag.wav")),
        make_pcm_source(preload("res://plastic crinkle.wav")),
        make_pcm_source(preload("res://plastic crunch.wav")),
        make_pcm_source(preload("res://tambourine.wav")),
    ]
    var pcm_source_custom = [Vector2()]
    var pcm_sample_loop = 1.0
    var pcm_source = 0.0
    var pcm_volume = 0.0
    var pcm_offset = 0.0
    var pcm_cutoff = 0.0
    var pcm_rate = 16.0
    var pcm_noise_cycle = 1024
    func _pcm(cursor):
        #if pcm_source == null or pcm_source.size() == 0:
        cursor = int(cursor*pcm_rate + pcm_offset*sample_rate)
        if int(pcm_source) == 0:
            seed(cursor % int(pcm_noise_cycle))
            var n = randf() * 2.0 - 1.0
            return n
        elif int(pcm_source) <= pcm_sources.size():
            var source = pcm_sources[int(pcm_source)-1]
            var size = source.size()
            if pcm_cutoff > 0.0:
                size = min(size, int(pcm_cutoff * sample_rate))
            if cursor >= size:
                if pcm_sample_loop != 0.0:
                    cursor = cursor % size
                else:
                    return 0.0
            return source[cursor]
        elif int(pcm_source) == pcm_sources.size()+1:
            var source = pcm_source_custom
            var size = source.size()
            if pcm_cutoff > 0.0:
                size = min(size, int(pcm_cutoff * sample_rate))
            if cursor >= size:
                if pcm_sample_loop != 0.0:
                    cursor = cursor % size
                else:
                    return 0.0
            return source[cursor]
        else:
            return 0.0
    
    var time_limit = 5.0
    
    var sin_volume = 0.0
    var sin_detune = 0.0
    
    var tri_volume = 0.0
    var tri_stages = 16.0
    var tri_detune = 0.0
    
    var square_volume = 0.5
    var square_width = 0.5
    var square_detune = 0.0
    
    var saw_volume = 0.0
    var saw_exponent = 1.0
    var saw_detune = 0.0
    
    func update_filters():
        var c = samples.size()-1
        
        if delay_wet_amount != 0.0:
            delay.push_sample(samples[c])
            samples[c] = delay.pop_sample()
        elif delay_dry_amount != 1.0:
            samples[c] *= delay_dry_amount
        
        if reverb_wet_amount != 0.0:
            reverb.push_sample(samples[c])
            samples[c] = reverb.pop_sample()
        elif reverb_dry_amount != 1.0:
            samples[c] *= reverb_dry_amount
        
        if lowpass_frequency < 22050.0:
            lowpass.push_sample(samples[c])
            samples[c] = lowpass.pop_sample()
        
        if highpass_frequency > 20.0:
            highpass.push_sample(samples[c])
            samples[c] = highpass.pop_sample()
        
        if ringmod_frequency > 0.0 and ringmod_amount != 0.0:
            ringmod.push_sample(samples[c])
            samples[c] = ringmod.pop_sample()
        
        if flanger_wet_amount != 0.0:
            flanger.push_sample(samples[c])
            samples[c] = flanger.pop_sample()
        elif flanger_dry_amount != 1.0:
            samples[c] *= flanger_dry_amount
        
        if (chorus_wet_amount != 0.0 and chorus_voices > 0):
            chorus.push_sample(samples[c])
            samples[c] = chorus.pop_sample()
        elif chorus_dry_amount != 1.0:
            samples[c] *= chorus_dry_amount
        
        if waveshaper.mix != 0.0:
            waveshaper.push_sample(samples[c])
            samples[c] = waveshaper.pop_sample()
        
        if true:
            limiter.push_sample(samples[c])
            samples[c] = limiter.pop_sample()
    
    var delay_time = 0.25
    var delay_decay = 0.2
    var delay_stereo_offset = -0.02
    var delay_wet_amount = 0.0
    var delay_dry_amount = 1.0
    var delay_diffusion = 0
    var delay_diffusion_ratio = 0.5
    
    var delay = Delay.new(sample_rate, delay_time, delay_decay)
    
    
    var reverb_dry_amount = 1.0
    var reverb_wet_amount = 0.0
    var reverb_decay = 0.5
    var reverb = Reverb.new(sample_rate, reverb_decay)
    
    var lowpass_frequency = 22050.0
    var lowpass = Lowpass.new(sample_rate, lowpass_frequency)
    
    var highpass_frequency = 20.0
    var highpass = Highpass.new(sample_rate, highpass_frequency)
    
    var ringmod_frequency = 0.25
    var ringmod_phase = 0.0
    var ringmod_amount = 0.0
    var ringmod = Ringmod.new(sample_rate, ringmod_frequency, ringmod_phase, ringmod_amount)
    
    var limiter_pre_gain = 1.0
    var limiter_lookahead = 0.001
    var limiter_attack = 0.001
    var limiter_sustain = 0.04
    var limiter_release = 0.04
    var limiter_threshold = 1.0
    var limiter = Limiter.new(sample_rate, limiter_pre_gain, limiter_lookahead, limiter_attack, limiter_sustain, limiter_release, limiter_threshold)
    
    var flanger_min_depth = 0.002
    var flanger_max_depth = 0.005
    var flanger_cycle_time = 2.0
    var flanger_wet_amount = 0.0
    var flanger_dry_amount = 1.0
    var flanger_feedback = 0.5
    
    var flanger = Flanger.new(sample_rate, flanger_min_depth, flanger_max_depth, flanger_cycle_time)
    
    var chorus_depth = 0.002
    var chorus_rate = 5.0
    var chorus_wet_amount = 0.0
    var chorus_dry_amount = 1.0
    var chorus_voices = 3
    var chorus = Chorus.new(sample_rate, chorus_depth, chorus_rate, chorus_wet_amount, chorus_dry_amount, chorus_voices)
    
    var waveshaper_pre_gain = 1.0
    var waveshaper_exponent = 1.0
    var waveshaper_clip = 1.0
    var waveshaper_clip_mode = 0 # 0: normal clipping. 1: "bounce" clipping. 2: wrapping
    var waveshaper_quantization = 0 # quantize to number of steps
    var waveshaper_mix = 0.0
    
    var waveshaper = Waveshaper.new(sample_rate, waveshaper_pre_gain, waveshaper_exponent, waveshaper_clip, waveshaper_clip_mode, waveshaper_quantization, waveshaper_mix)
    
    
    var gen_nonce = 0
    func generate():
        gen_nonce += 1
        var self_nonce = gen_nonce
        parent.find_node("Regen").disabled = true
        #pcm_source = make_pcm_source(preload("res://tambourine.wav"))
        #pcm_source = make_pcm_source(preload("res://paper bag.wav"))
        samples = PoolVector2Array()
        restart()
        
        var aa = oversampling
        var break_limit = 0.2
        var silence_count = 0
        var silence_limit = 1.0/32768.0
        
        var start_time = OS.get_ticks_msec()
        for _i in range(sample_rate*time_limit):
            if abs(OS.get_ticks_msec() - start_time) > 10:
                yield(parent.get_tree(), "idle_frame")
                if gen_nonce != self_nonce:
                    return
                start_time = OS.get_ticks_msec()
            update_envelope(gen_time)
            var old_time = gen_time
            var next = Vector2.ZERO
            var current_freq = freq * semitones_to_factor(freq_offset_lfo + freq_offset_sweep + freq_offset_step)
            for _j in range(aa):
                gen_cursor += current_freq/sample_rate/aa*2.0
                if sin_volume != 0.0:
                    var f = semitones_to_factor(sin_detune)    if sin_detune    != 0.0 else 1.0
                    next += Vector2.ONE * sin_volume    * _sin   (gen_cursor * f)
                if tri_volume != 0.0:
                    var f = semitones_to_factor(tri_detune)    if tri_detune    != 0.0 else 1.0
                    next += Vector2.ONE * tri_volume    * _tri   (gen_cursor * f, tri_stages)
                if square_volume != 0.0:
                    var f = semitones_to_factor(square_detune) if square_detune != 0.0 else 1.0
                    next += Vector2.ONE * square_volume * _square(gen_cursor * f, square_width)
                if saw_volume != 0.0:
                    var f = semitones_to_factor(saw_detune)    if saw_detune    != 0.0 else 1.0
                    next += Vector2.ONE * saw_volume    * _saw   (gen_cursor * f, saw_exponent)
                if pcm_volume != 0.0:
                    next += Vector2.ONE * pcm_volume    * _pcm   (gen_cursor)
            var sample = next * 0.5 / aa * envelope
            samples.push_back(sample)
            gen_time += 1/sample_rate
            if update_events(old_time):
                break
            update_filters()
            sample = samples[samples.size()-1]
            
            if abs(sample.x) < silence_limit and abs(sample.y) < silence_limit:
                silence_count += 1
            else:
                #print("resetting silence count")
                silence_count = 0
            if silence_count/sample_rate > break_limit:
                break
        playback_cursor = 0
        parent.find_node("Regen").disabled = false
        emit_signal("generation_complete")
    
    var step_time = 0.0
    var step_semitones = 4.0
    var step_semitones_stagger = -1.0
    var step_retrigger = 1.0
    var step_loop = 0.0
    
    var freq_lfo_rate = 0.0
    var freq_lfo_strength = 0.0
    var freq_lfo_shape = 0.0
    
    var freq_sweep_rate = 0.0 # semitones per second
    var freq_sweep_delta = 0.0 # semitones per second per second
    
    ## FIXME: AHR volume isn't implemented correctly
    # (note to self: use points 0, 1/2, and 3, not 123 or 234)
    var attack = 0.0
    var attack_exponent = 1.0
    var attack_volume = 1.0
    var hold = 0.2
    var hold_volume = 1.0
    var release = 0.8
    var release_exponent = 4.0
    var release_volume = 1.0
    
    var envelope = 1.0
    
    func update_envelope(time):
        if time < attack and attack > 0.0:
            envelope = time/attack
            envelope = pow(envelope, attack_exponent) * attack_volume
        elif time - attack < hold:
            envelope = hold_volume
        elif time - attack - hold < release:
            if release > 0.0:
                envelope = max(0, 1.0 - ((time - attack - hold)/release))
                if release_exponent != 1.0:
                    envelope = pow(envelope, release_exponent)
                envelope *= release_volume
            else:
                envelope = 0.0
        else:
            envelope = 0.0
    
    func update_events(old_time):
        if step_time > 0.0:
            var trigger_time = gen_time
            var step_semitones = self.step_semitones
            var step_semitones_stagger = self.step_semitones_stagger
            if step_retrigger != 0.0:
                if fmod(old_time, step_time*step_loop) > fmod(trigger_time, step_time*step_loop):
                    freq_offset_step = 0
                while step_loop > 0.0 and trigger_time > step_time*step_loop:
                    trigger_time -= step_time*step_loop
                    old_time -= step_time*step_loop
                while old_time > step_time:
                    old_time -= step_time
                    trigger_time -= step_time
                    step_semitones += step_semitones_stagger
                    step_semitones_stagger = -step_semitones_stagger
            if old_time < step_time and trigger_time >= step_time:
                freq_offset_step += step_semitones
        
        if freq_lfo_strength != 0 and freq_lfo_rate != 0:
            var t = gen_time * freq_lfo_rate * 2.0
            if freq_lfo_shape == 0.0:
                freq_offset_lfo = _tri(t)
            elif freq_lfo_shape == 1.0:
                freq_offset_lfo = _square(t)
            elif freq_lfo_shape == 2.0:
                freq_offset_lfo = _square(t, 0.25, true)
            elif freq_lfo_shape == 3.0:
                freq_offset_lfo = _saw(t)
            elif freq_lfo_shape == 4.0:
                freq_offset_lfo = _saw(t, 3)
            elif freq_lfo_shape == 5.0:
                if int(pcm_source) == 0:
                    freq_offset_lfo = _pcm(t / pcm_rate)
                else:
                    freq_offset_lfo = _pcm(t / pcm_rate)
                    freq_offset_lfo = freq_offset_lfo.x + freq_offset_lfo.y
                if randi() % 10000 == 0:
                    print(freq_offset_lfo)
            freq_offset_lfo *= freq_lfo_strength
        else:
            freq_offset_lfo = 0.0
        
        var sweep_offset = freq_sweep_delta * gen_time
        freq_offset_sweep += (freq_sweep_rate + sweep_offset) * 1.0/sample_rate
        #return true
        
    func restart():
        playback_cursor = 0
        freq_offset_lfo = 0.0 # semitones
        freq_offset_sweep = 0.0 # semitones
        freq_offset_step = 0.0 # semitones
        gen_time = 0.0
        gen_cursor = 0.0
        
        last_sq_cursor = 0.0
        last_saw_cursor = 0.0
        
        delay = Delay.new(sample_rate, delay_time, delay_decay)
        delay.stereo_offset = delay_stereo_offset
        delay.wet_amount = delay_wet_amount
        delay.dry_amount = delay_dry_amount
        delay.diffusion = delay_diffusion
        delay.diffusion_ratio = delay_diffusion_ratio
        
        reverb = Reverb.new(sample_rate, reverb_decay)
        reverb.wet_amount = reverb_wet_amount
        reverb.dry_amount = reverb_dry_amount
        
        lowpass = Lowpass.new(sample_rate, lowpass_frequency)
        highpass = Highpass.new(sample_rate, highpass_frequency)
        ringmod = Ringmod.new(sample_rate, ringmod_frequency, ringmod_phase, ringmod_amount)
        
        flanger = Flanger.new(sample_rate, flanger_min_depth, flanger_max_depth, flanger_cycle_time)
        flanger.wet_amount = flanger_wet_amount
        flanger.dry_amount = flanger_dry_amount
        flanger.feedback = flanger_feedback
        
        chorus = Chorus.new(sample_rate, chorus_depth, chorus_rate, chorus_wet_amount, chorus_dry_amount, chorus_voices)
        
        waveshaper = Waveshaper.new(sample_rate,
            waveshaper_pre_gain,
            waveshaper_exponent,
            waveshaper_clip,
            waveshaper_clip_mode,
            waveshaper_quantization,
            waveshaper_mix
        )
        
        limiter = Limiter.new(sample_rate,
            limiter_pre_gain,
            limiter_lookahead,
            limiter_attack,
            limiter_sustain,
            limiter_release,
            limiter_threshold
        )
    
    func pull_sample():
        playback_cursor = max(0, playback_cursor)
        if playback_cursor < samples.size():
            playback_cursor += 1
            return samples[playback_cursor-1]
        else:
            return Vector2.ZERO
    
    signal generation_complete

func set_value(key, value):
    sliders[key].value = value

func randomize_value(key, _range : Array):
    var slider : Range = sliders[key]
    if slider.max_value <= slider.min_value:
        return
    
    # assigning to .value applies the slider's own limit, then triggers the signal that updates the generator
    if !slider.exp_edit or slider.min_value <= 0.0:
        slider.value = rand_range(_range[0], _range[1])
    else:
        var step = max(0.001, slider.step)
        _range[0] = max(step, _range[0])
        _range[1] = max(step, _range[1])
        slider.value = exp(rand_range(log(_range[0]), log(_range[1])))

func reset_all_values():
    for key in default_values.keys():
        if key in ["oversampling", "time_limit"]:
            continue
        set_value(key, default_values[key])
        pass

func random_choice(array : Array):
    return array[randi() % array.size()]

func random_pickup():
    seed(OS.get_ticks_usec() ^ hash("awfiei"))
    reset_all_values()
    set_value("square_volume", 0.0)
    
    var which = random_choice(["square", "tri"])
    set_value("%s_volume" % which, 1.0)
    
    randomize_value("freq", [400.0, 2400.0])
    randomize_value("hold", [0.0, 0.1])
    randomize_value("release", [0.1, 0.5])
    
    if randi() % 2:
        randomize_value("step_time", [0.05, 0.1])
        randomize_value("step_semitones", [1.0, 7.0])
        set_value("step_retrigger", 0)
    
    generator.generate()

func random_laser():
    seed(OS.get_ticks_usec() ^ hash("awfiei"))
    reset_all_values()
    set_value("square_volume", 0.0)
    
    randomize_value("freq", [400.0, 4400.0])
    randomize_value("hold", [0.2, 0.4])
    randomize_value("release", [0.05, 0.15])
    
    randomize_value("freq_sweep_rate", [-48.0, -256.0])
    
    var which = random_choice(["square", "tri", "saw", "sin"])
    set_value("%s_volume" % which, 1.0)
    if which == "square":
        randomize_value("square_width", [0.5, 0.8])
    
    set_value("highpass_frequency", 100)
    
    generator.generate()

func random_explosion(no_delay = false):
    seed(OS.get_ticks_usec() ^ hash("awfiei"))
    reset_all_values()
    set_value("square_volume", 0.0)
    
    randomize_value("freq", [40.0, 700.0])
    randomize_value("hold", [0.0, 0.5])
    randomize_value("release", [0.5, 1.5])
    
    if randi() % 4 > 0:
        randomize_value("freq_sweep_rate", [-256.0, 64.0])
    if generator.freq > 500.0:
        randomize_value("freq_sweep_rate", [-256.0, 0.0])
    
    if randi() % 3 == 0:
        randomize_value("freq_lfo_rate", [3.0, 6.0])
        randomize_value("freq_lfo_strength", [0.5, 12.0])
    if randi() % 3 == 0:
        set_value("ringmod_phase", 0.25)
        randomize_value("ringmod_frequency", [1.0, 4.0])
        randomize_value("ringmod_amount", [0.5, 1.0])
    
    if !no_delay and randi() % 3 == 0:
        randomize_value("delay_time", [0.2, 0.4])
        randomize_value("delay_wet_amount", [0.1, 0.4])
        randomize_value("delay_diffusion", [0.0, 4.0])
        set_value("delay_decay", 0.0)
        
    set_value("pcm_volume", 1.0)
    randomize_value("pcm_noise_cycle", [32.0, 65536])
    
    #set_value("highpass_frequency", 100)
    
    generator.generate()

func random_powerup():
    seed(OS.get_ticks_usec() ^ hash("awfiei"))
    reset_all_values()
    set_value("square_volume", 0.0)
    
    var which = random_choice(["square", "tri"])
    set_value("%s_volume" % which, 1.0)
    
    randomize_value("freq", [200.0, 2400.0])
    randomize_value("attack", [0.0, 0.02])
    randomize_value("hold", [0.3, 0.5])
    randomize_value("release", [0.5, 1.0])
    
    var slide_type = randi() % 3
    if slide_type == 0:
        randomize_value("step_time", [0.02, 0.05])
        randomize_value("step_semitones", [1.0, 7.0])
        set_value("step_semitones_stagger", 0.0)
        set_value("step_retrigger", 1.0)
        randomize_value("step_loop", [3.0, 6.0])
    elif slide_type == 1:
        randomize_value("freq_sweep_rate", [7.0, 128.0])
    else:
        randomize_value("freq_lfo_rate", [3.0, 20.0])
        randomize_value("freq_lfo_strength", [6.0, 24.0])
        set_value("freq_lfo_shape", 3)
        
    if slide_type != 2 and randi() % 2:
        randomize_value("freq_lfo_rate", [3.0, 6.0])
        randomize_value("freq_lfo_strength", [0.2, 2.0])
        
    if randi() % 4:
        set_value("delay_wet_amount", 0.2)
    if randi() % 4:
        set_value("chorus_wet_amount", 0.2)
    
    generator.generate()

func random_jump():
    seed(OS.get_ticks_usec() ^ 4136436345)
    reset_all_values()
    
    randomize_value("square_width", [0.5, 0.8])
    
    randomize_value("freq", [50.0, 400.0])
    randomize_value("hold", [0.1, 0.5])
    randomize_value("release", [0.1, 0.2])
    
    randomize_value("freq_sweep_rate", [15.0, 128.0])
    
    generator.generate()

func random_hit():
    seed(OS.get_ticks_usec() ^ 4136436345)
    if randi() % 4 == 0:
        random_explosion(true)
    else:
        random_laser()
    
    randomize_value("freq", [40.0, 800.0])
    randomize_value("hold", [0.05, 0.10])
    randomize_value("release", [0.15, 0.25])
    randomize_value("freq_sweep_rate", [-128.0, -512.0])

func random_blip():
    seed(OS.get_ticks_usec() ^ hash("awfiei"))
    reset_all_values()
    set_value("square_volume", 0.0)
    
    var which = random_choice(["square", "tri", "sin"])
    set_value("%s_volume" % which, 1.0)
    
    randomize_value("freq", [200.0, 2400.0])
    randomize_value("hold", [0.1, 0.2])
    randomize_value("release", [0.01, 0.05])
    
    if randi() % 2 == 0:
        randomize_value("step_time", [0.02, 0.085])
        randomize_value("step_semitones", [-1.0, -24.0])
        set_value("step_retrigger", 0)
    
    generator.generate()

onready var control_target : Node = $Scroll/Box/A/Scroller/Controls

func set_label_value(label : Label, value : float):
    if abs(value) == 0.0:
        label.text = "0.00"
    elif abs(value) < 0.1:
        label.text = "%.3f" % value
    elif abs(value) < 10:
        label.text = "%.2f" % value
    elif abs(value) < 1000:
        label.text = "%.1f" % value
    else:
        label.text = "%.0f" % value

func slider_update(value : float, _slider : Range, number : Label, name : String):
    set_label_value(number, value)
    generator.set(name, value)
    print(name)
    print(generator.get(name))
    pass

var default_values = {}
var sliders = {}

func add_slider(name : String, min_value, max_value):
    var value = generator.get(name)
    
    var label = Label.new()
    label.text = name.capitalize()
    
    var number = Label.new()
    set_label_value(number, value)
    number.size_flags_horizontal |= SIZE_EXPAND_FILL
    number.size_flags_stretch_ratio = 0.25
    number.align = Label.ALIGN_RIGHT
    number.clip_text = true
    
    var slider = HSlider.new()
    slider.name = name
    slider.min_value = min_value
    slider.max_value = max_value
    slider.step = 0.01
    slider.size_flags_horizontal |= SIZE_EXPAND_FILL
    slider.size_flags_stretch_ratio = 0.75
    slider.tick_count = 5
    slider.ticks_on_borders = true
    slider.add_to_group("Slider")
    sliders[name] = slider
    
    slider.connect("value_changed", self, "slider_update", [slider, number, name])
    slider.value = value
    
    var container = HSplitContainer.new()
    container.dragger_visibility = container.DRAGGER_HIDDEN
    container.add_child(number)
    container.add_child(slider)
    
    control_target.add_child(label)
    control_target.add_child(container)
    return slider

func add_separator():
    var separator = HSeparator.new()
    control_target.add_child(separator)

func add_controls():
    var slider : Slider
    slider = add_slider("freq", 20, 22050)
    slider.exp_edit = true
    slider.step = 0.5
    add_separator()
    add_slider("square_volume", -1.0, 1.0)
    add_slider("square_detune", -48.0, 48.0)
    add_slider("square_width", 0.0, 1.0)
    add_separator()
    add_slider("tri_volume", -1.0, 1.0)
    add_slider("tri_detune", -48.0, 48.0)
    slider = add_slider("tri_stages", 0.0, 32.0)
    slider.step = 1
    add_separator()
    add_slider("saw_volume", -1.0, 1.0)
    add_slider("saw_detune", -48.0, 48.0)
    slider = add_slider("saw_exponent", 0.01, 16.0)
    slider.exp_edit = true
    add_separator()
    add_slider("sin_volume", -1.0, 1.0)
    add_slider("sin_detune", -48.0, 48.0)
    add_separator()
    add_slider("pcm_volume", -1.0, 1.0)
    add_slider("pcm_offset", 0.0, 5.0)
    add_slider("pcm_cutoff", 0.0, 15.0)
    add_slider("pcm_rate", 0.01, 100.0).exp_edit = true
    slider = add_slider("pcm_noise_cycle", 2, pow(2, 16))
    slider.exp_edit = true
    slider.step = 1
    add_slider("pcm_source", 0, 5).step = 1
    add_slider("pcm_sample_loop", 0, 1).step = 1
    
    control_target = $Scroll/Box/B/Scroller/Controls
    
    add_slider("step_time", 0.0, 5.0)
    add_slider("step_semitones", -48, 48)
    add_slider("step_semitones_stagger", -48, 48)
    add_slider("step_retrigger", 0, 1).step = 1
    add_slider("step_loop", 0, 16).step = 1
    add_separator()
    add_slider("freq_lfo_rate", 0.0, 50)
    add_slider("freq_lfo_strength", -12, 12)
    add_slider("freq_lfo_shape", 0, 5).step = 1
    add_separator()
    add_slider("freq_sweep_rate", -12*32, 12*32).step = 1
    add_slider("freq_sweep_delta", -12*32, 12*32).step = 1
    add_separator()
    add_slider("ringmod_frequency", 0.01, 22050.0).exp_edit = true
    add_slider("ringmod_phase", 0.0, 1.0)
    add_slider("ringmod_amount", -2.0, 2.0)
    
    control_target = $Scroll/Box/C/Scroller/Controls
    
    slider = add_slider("time_limit", 0.1, 50.0)
    slider.exp_edit = true
    add_separator()
    add_slider("attack", 0.0, 5.0)
    slider = add_slider("attack_exponent", 0.1, 10.0)
    slider.exp_edit = true
    add_slider("attack_volume", -1.0, 1.0)
    add_separator()
    add_slider("hold", 0.0, 10.0)
    add_slider("hold_volume", -1.0, 1.0)
    add_separator()
    add_slider("release", 0.01, 20.0).exp_edit = true
    slider = add_slider("release_exponent", 0.1, 10.0)
    slider.exp_edit = true
    add_slider("release_volume", -1.0, 1.0)
    add_separator()
    add_slider("limiter_pre_gain", 0.05, 20)
    add_slider("limiter_lookahead", 0.0, 0.01).step = 0.001
    add_slider("limiter_attack", 0.0, 0.01).step = 0.001
    add_slider("limiter_sustain", 0.0, 0.5)
    add_slider("limiter_release", 0.0, 0.5)
    add_slider("limiter_threshold", 0.0, 1.0)
    
    control_target = $Scroll/Box/D/Scroller/Controls
    add_slider("oversampling", 1, 8.0).step = 1
    add_separator()
    add_slider("delay_time", 0.001, 4.0).step = 0.001
    add_slider("delay_decay", 0.0, 2.0)
    add_slider("delay_stereo_offset", -1.0, 1.0).step = 0.001
    add_slider("delay_wet_amount", -1.0, 1.0)
    add_slider("delay_dry_amount", -1.0, 1.0)
    add_slider("delay_diffusion", 0.0, 8.0).step = 1
    add_slider("delay_diffusion_ratio", 0.0, 1.0)
    add_separator()
    add_slider("reverb_decay", 0.0, 0.9)
    add_slider("reverb_wet_amount", -8.0, 8.0)
    add_slider("reverb_dry_amount", -1.0, 1.0)
    add_separator()
    add_slider("lowpass_frequency", 20.0, 22050.0).exp_edit = true
    add_separator()
    add_slider("highpass_frequency", 20.0, 22050.0).exp_edit = true
    add_separator()
    add_slider("flanger_min_depth", 0.0, 0.5)
    add_slider("flanger_max_depth", 0.0, 0.5)
    add_slider("flanger_cycle_time", 0.01, 20).exp_edit = true
    add_slider("flanger_wet_amount", -1.0, 1.0)
    add_slider("flanger_dry_amount", -1.0, 1.0)
    add_slider("flanger_feedback", 0.0, 1.0)
    add_separator()
    add_slider("chorus_depth", 0.0, 0.05).step = 0.001
    add_slider("chorus_rate", 0.0, 50)
    add_slider("chorus_wet_amount", 0.0, 1.0)
    add_slider("chorus_dry_amount", 0.0, 1.0)
    add_slider("chorus_voices", 0, 8).step = 1
    add_separator()
    add_slider("waveshaper_pre_gain", 0.0, 16.0)
    add_slider("waveshaper_exponent", 0.1, 10.0).exp_edit = true
    add_slider("waveshaper_clip", 0.0, 8.0)
    add_slider("waveshaper_clip_mode", 0, 2).step = 1
    add_slider("waveshaper_quantization", 0, 256).step = 1
    add_slider("waveshaper_mix", 0.0, 1.0)
    
    yield(get_tree(), "idle_frame")
    for _slider in get_tree().get_nodes_in_group("Slider"):
        print(_slider.name)
        var value = generator.get(_slider.name)
        default_values[_slider.name] = value
        _slider.value = value
        pass
    #add_separator()

func _on_files_dropped(files : PoolStringArray, _screen : int):
    #var music = AudioStreamPlayer.new()
    var audio_loader = AudioLoader.new()
    var stream = audio_loader.loadfile(files[0])
    if not stream is AudioStreamSample:
        return
    generator.pcm_source_custom = generator.make_pcm_source(stream)
    generator.pcm_source = generator.pcm_sources.size()+1
    $Scroll/Box/ScrollerA/Controls.find_node("pcm_source", true, false).value = generator.pcm_source

var fname
var fname_bare
func save():
    print("asdF")
    var dir = Directory.new()
    dir.make_dir("sfx_output")
    
    var timestamp = OS.get_system_time_msecs()
    fname = "sfx_output/sfx_%s.wav" % timestamp
    fname_bare = "sfx_%s.wav" % timestamp
    var bytes = StreamPeerBuffer.new()
    for vec in generator.samples:
        vec.x = clamp(vec.x, -1.0, 1.0)
        vec.y = clamp(vec.y, -1.0, 1.0)
        bytes.put_16(vec.x*32767.0)
        bytes.put_16(vec.y*32767.0)
    bytes.seek(0)
    var audio = AudioStreamSample.new()
    audio.format = audio.FORMAT_16_BITS
    audio.stereo = true
    print(bytes.get_available_bytes())
    audio.data = bytes.data_array
    
    if OS.get_name() == "HTML5":
        fname = "user://" + fname_bare
    
    audio.save_to_wav(fname)
    var file = File.new()
    file.open(fname, File.READ)
    
    JavaScript.download_buffer(file.get_buffer(file.get_len()), fname_bare, "audio/wave")
    pass

func update_player():
    $Player.stop()
    
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    
    var bytes = StreamPeerBuffer.new()
    for vec in generator.samples:
        vec.x = clamp(vec.x, -1.0, 1.0)
        vec.y = clamp(vec.y, -1.0, 1.0)
        bytes.put_16(vec.x*32767.0)
        bytes.put_16(vec.y*32767.0)
    bytes.seek(0)
    var audio = AudioStreamSample.new()
    audio.format = audio.FORMAT_16_BITS
    audio.stereo = true
    audio.data = bytes.data_array
    $Player.stream = audio
    
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    
    $Player.play()

var generator : Reference
var playback
var ready = false
func _ready():
    var _unused = get_tree().connect("files_dropped", self, "_on_files_dropped")
    
    generator = Generator.new()
    generator.parent = self
    add_controls()
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    _unused = generator.connect("generation_complete", self, "update_player")
    generator.generate()
    
    #$Player.stream = AudioStreamGenerator.new()
    #$Player.stream.mix_rate = generator.sample_rate
    #playback = $Player.get_stream_playback()
    #$Player.play()
    
    ready = true
    
    _unused = $Buttons/Regen.connect("pressed", generator, "generate")
    _unused = $Buttons/Save.connect("pressed", self, "save")
    
    _unused = $Buttons2/Pickup.connect("pressed", self, "random_pickup")
    _unused = $Buttons2/Laser.connect("pressed", self, "random_laser")
    _unused = $Buttons2/Explosion.connect("pressed", self, "random_explosion")
    _unused = $Buttons2/Powerup.connect("pressed", self, "random_powerup")
    _unused = $Buttons2/Hit.connect("pressed", self, "random_hit")
    _unused = $Buttons2/Jump.connect("pressed", self, "random_jump")
    _unused = $Buttons2/Blip.connect("pressed", self, "random_blip")
    
    yield(get_tree().create_timer(generator.samples.size() / generator.sample_rate + 0.25), "timeout")
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    
    
    

func _process(_delta):
    if !ready:
        return
    #for _i in range(playback.get_frames_available()):
    #    playback.push_frame(generator.pull_sample())

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#    pass
