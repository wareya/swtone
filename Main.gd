extends Control

class Filter extends Reference:
    var sample_rate = 1.0
    func _init(sample_rate):
        self.sample_rate = sample_rate
    func push_sample(x):
        pass
    func pop_sample():
        pass

class Delay extends Filter:
    var wet = []
    var wet_amount = 0.25
    var dry_amount = 1.0
    var dry = Vector2.ZERO
    var decay = 0.2
    var cursor = 0
    var stereo_offset = -0.01
    func _init(sample_rate, time, decay).(sample_rate):
        self.decay = decay
        wet = []
        for i in range(time*sample_rate):
            wet.push_back(Vector2.ZERO)
    func push_sample(x):
        wet[cursor] *= decay
        wet[cursor] += x
        dry = x
        cursor = (cursor + 1) % wet.size()
    func pop_sample():
        var _wet = wet[cursor]
        if stereo_offset > 0.0:
            var c2 = int(cursor+stereo_offset*sample_rate)
            c2 = c2 % wet.size()
            _wet.y = wet[c2].y
        elif stereo_offset < 0.0:
            var c2 = int(cursor-stereo_offset*sample_rate)
            c2 = c2 % wet.size()
            _wet.x = wet[c2].x
        return dry * dry_amount + _wet * wet_amount
    

class Generator extends Reference:
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
    
    func _square(cursor, width = 0.5):
        var x = fmod(cursor, 2.0)/2.0
        var out = -1.0 if x < width else 1.0
        var dc_bias = (width - 0.5) * 2
        return out + dc_bias
    
    func _saw(cursor, exponent = 1.0):
        var n = fmod(cursor, 2.0)-1.0
        var out = pow(abs(n), exponent)*sign(n)
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
    var pcm_volume = 0.5
    var pcm_offset = 0.0
    var pcm_rate = 16.0
    var pcm_noise_cycle = 1024
    func _pcm(cursor, exponent = 1.0):
        #if pcm_source == null or pcm_source.size() == 0:
        cursor = int(cursor*pcm_rate + pcm_offset*sample_rate)
        if int(pcm_source) == 0:
            seed(cursor % int(pcm_noise_cycle))
            var n = randf() * 2.0 - 1.0
            return n
        elif int(pcm_source) <= pcm_sources.size():
            var source = pcm_sources[int(pcm_source)-1]
            if cursor >= source.size():
                if pcm_sample_loop != 0.0:
                    cursor = cursor % source.size()
                else:
                    return 0.0
            return source[cursor]
        elif int(pcm_source) == pcm_sources.size()+1:
            var source = pcm_source_custom
            if cursor >= source.size():
                if pcm_sample_loop != 0.0:
                    cursor = cursor % source.size()
                else:
                    return 0.0
            return source[cursor]
        else:
            return 0.0
    
    var time_limit = 5.0
    
    var sin_volume = 0.0
    
    var tri_volume = 0.0
    var tri_stages = 16.0
    
    var square_volume = 0.0
    var square_width = 0.5
    
    var saw_volume = 0.0
    var saw_exponent= 1.0
    
    func update_filters():
        delay.push_sample(samples[samples.size()-1])
        samples[samples.size()-1] = delay.pop_sample()
    
    var delay_time = 0.25
    var delay_decay = 0.2
    var delay_stereo_offset = -0.005
    var delay_wet_amount = 0.0
    var delay_dry_amount = 1.0
    
    var delay = Delay.new(sample_rate, delay_time, delay_decay)
    
    func generate():
        #pcm_source = make_pcm_source(preload("res://tambourine.wav"))
        #pcm_source = make_pcm_source(preload("res://paper bag.wav"))
        samples = PoolVector2Array()
        restart()
        
        var aa = oversampling
        var break_limit = 0.1
        var silence_count = 0
        var silence_limit = 1.0/32768.0
        
        for x in range(sample_rate*time_limit):
            update_envelope(gen_time)
            var old_time = gen_time
            var next = Vector2.ZERO
            var current_freq = freq * semitones_to_factor(freq_offset_lfo + freq_offset_sweep + freq_offset_step)
            for i in range(aa):
                gen_cursor += (current_freq)/sample_rate/aa
                if sin_volume != 0.0:
                    next += Vector2.ONE * sin_volume    * _sin   (gen_cursor)
                if tri_volume != 0.0:
                    next += Vector2.ONE * tri_volume    * _tri   (gen_cursor, tri_stages)
                if square_volume != 0.0:
                    next += Vector2.ONE * square_volume * _square(gen_cursor, square_width)
                if saw_volume != 0.0:
                    next += Vector2.ONE * saw_volume    * _saw   (gen_cursor, saw_exponent)
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
        emit_signal("generation_complete")
    
    var step_time = 0.0
    var step_semitones = 4
    var step_semitones_stagger = -1
    var step_retrigger = 1.0
    
    var freq_lfo_rate = 0.0
    var freq_lfo_strength = 0.0
    
    var freq_sweep_rate = 0.0 # semitones per second
    var freq_sweep_delta = 0.0 # semitones per second per second
    
    var attack = 0.0
    var attack_exponent = 1.0
    var attack_volume = 1.0
    var hold = 0.0
    var hold_volume = 1.0
    var release = 1.5
    var release_exponent = 5.0
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
                envelope = pow(envelope, release_exponent) * release_volume
            else:
                envelope = 0.0
    
    func update_events(old_time):
        var trigger_time = gen_time
        var step_semitones = self.step_semitones
        var step_semitones_stagger = self.step_semitones_stagger
        if step_retrigger != 0.0 and step_time > 0.0:
            while old_time > step_time:
                old_time -= step_time
                trigger_time -= step_time
                step_semitones += step_semitones_stagger
                step_semitones_stagger = -step_semitones_stagger
        if step_time > 0.0 and old_time < step_time and trigger_time >= step_time:
            freq_offset_step += step_semitones
        
        var f = _tri(gen_time*freq_lfo_rate*2.0) * freq_lfo_strength
        freq_offset_lfo = f
        
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
        
        delay = Delay.new(sample_rate, delay_time, delay_decay)
        delay.stereo_offset = delay_stereo_offset
        delay.wet_amount = delay_wet_amount
        delay.dry_amount = delay_dry_amount
    
    func pull_sample():
        playback_cursor = max(0, playback_cursor)
        if playback_cursor < samples.size():
            playback_cursor += 1
            return samples[playback_cursor-1]
        else:
            return Vector2.ZERO
    
    signal generation_complete

onready var control_target : Node = $ScrollerA/Controls

func set_label_value(label : Label, value : float):
    if abs(value) < 10:
        label.text = "%.2f" % value
    elif abs(value) < 1000:
        label.text = "%.1f" % value
    else:
        label.text = "%.0f" % value

func slider_update(value : float, slider : Range, number : Label, name : String):
    set_label_value(number, value)
    generator.set(name, value)
    print(name)
    print(generator.get(name))
    pass

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
    
    slider.connect("value_changed", self, "slider_update", [slider, number, name])
    slider.value = value
    
    var container = HSplitContainer.new()
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
    add_slider("square_width", 0.0, 1.0)
    add_separator()
    add_slider("tri_volume", -1.0, 1.0)
    slider = add_slider("tri_stages", 0.0, 32.0)
    slider.step = 1
    add_separator()
    add_slider("saw_volume", -1.0, 1.0)
    slider = add_slider("saw_exponent", 0.01, 16.0)
    slider.exp_edit = true
    add_separator()
    add_slider("sin_volume", -1.0, 1.0)
    add_separator()
    add_slider("pcm_volume", -1.0, 1.0)
    add_slider("pcm_offset", 0.0, 5.0)
    add_slider("pcm_rate", 0.01, 100.0).exp_edit = true
    slider = add_slider("pcm_noise_cycle", 2, pow(2, 16))
    slider.exp_edit = true
    slider.step = 1
    add_slider("pcm_source", 0, 5).step = 1
    add_slider("pcm_sample_loop", 0, 1).step = 1
    
    control_target = $ScrollerB/Controls
    
    add_slider("step_time", 0.0, 5.0)
    add_slider("step_semitones", -48, 48)
    add_slider("step_semitones_stagger", -48, 48)
    add_slider("step_retrigger", 0, 1).step = 1
    add_separator()
    add_slider("freq_lfo_rate", 0.0, 50)
    add_slider("freq_lfo_strength", -12, 12)
    add_separator()
    add_slider("freq_sweep_rate", -12*32, 12*32).step = 1
    add_slider("freq_sweep_delta", -12*32, 12*32).step = 1
    
    control_target = $ScrollerC/Controls
    
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
    
    control_target = $ScrollerD/Controls
    add_slider("oversampling", 1, 8.0).step = 1
    add_separator()
    add_slider("delay_time", 0.01, 4.0)
    add_slider("delay_decay", 0.0, 2.0)
    add_slider("delay_stereo_offset", -1.0, 1.0)
    add_slider("delay_wet_amount", -1.0, 1.0)
    add_slider("delay_dry_amount", -1.0, 1.0)

func _on_files_dropped(files : PoolStringArray, screen : int):
    var want = files[0]
    
    var music = AudioStreamPlayer.new()
    var audio_loader = AudioLoader.new()
    var stream = audio_loader.loadfile(files[0])
    if not stream is AudioStreamSample:
        return
    generator.pcm_source_custom = generator.make_pcm_source(stream)
    generator.pcm_source = generator.pcm_sources.size()+1
    $ScrollerA/Controls.find_node("pcm_source", true, false).value = generator.pcm_source

var fname
var fname_bare
func save():
    print("asdF")
    var dir = Directory.new()
    dir.make_dir("sfx_output")
    
    var timestamp = OS.get_unix_time()
    fname = "sfx_output/sfx_%s.wav" % timestamp
    fname_bare = "sfx_%s.wav" % timestamp
    var bytes = StreamPeerBuffer.new()
    for vec in generator.samples:
        bytes.put_16(vec.x*32768.0)
        bytes.put_16(vec.y*32768.0)
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
        bytes.put_16(vec.x*32768.0)
        bytes.put_16(vec.y*32768.0)
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
    get_tree().connect("files_dropped", self, "_on_files_dropped")
    
    generator = Generator.new()
    add_controls()
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    yield(get_tree(), "idle_frame")
    generator.connect("generation_complete", self, "update_player")
    generator.generate()
    
    #$Player.stream = AudioStreamGenerator.new()
    #$Player.stream.mix_rate = generator.sample_rate
    #playback = $Player.get_stream_playback()
    #$Player.play()
    
    ready = true
    
    $Buttons/Regen.connect("pressed", generator, "generate")
    $Buttons/Save.connect("pressed", self, "save")
    
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
