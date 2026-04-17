// Shared avatar player for the landing page and sign mode.

window.SAMVAADAvatar = (function () {

    const api = {};

    // ── Internal state ────────────────────────────────────────────────────────
    let _renderer, _scene, _camera, _mixer, _clock;
    let _model        = null;
    let _modelReady   = false;
    let _modelLoading = false;
    let _baseAssetManager = null;
    let _assetsReady  = false;
    // Clip cache: base name → AnimationClip | null (null = file confirmed missing)
    const _clipCache   = {};
    // File-exists cache: full path → true | false
    const _existsCache = {};

    let _queue         = [];
    let _isPlaying     = false;
    let _currentAction = null;
    let _idleAction    = null;
    let _playbackToken = 0;

    let _onSignChange  = null;
    let _animFolder    = "animations/";
    let _container     = null;
    let _loadingEl     = null;
    let _initialized   = false;
    let _speed         = 1.0;
    let _rotationY     = 0;

    // Keys that use the LEFT hand (numbers + YES/NO)
    const LEFT_HAND_KEYS = new Set([
        '1','2','3','4','5','6','7','8','9','10','yes','no'
    ]);

    // Transition clip base-names
    const T = {
        IDLE        : "idle",
        RAISE_RIGHT : "right_hand_raise",
        LOWER_RIGHT : "lower_right",
        RAISE_LEFT  : "raise_left",
        LOWER_LEFT  : "lower_left",
        SPACE       : "space",        // between-word gap gesture
    };

    // All transition keys as a Set for fast lookup
    const TRANSITION_KEYS = new Set(Object.values(T));

    const CROSSFADE_DURATION = 0.15;  // seconds
    const MIN_SIGN_MS        = 600;   // minimum ms any sign is shown
    const IDLE_INTRO_MS      = 2000;  // how long idle plays before the intro sequence
    const CLIP_SETTLE_MS     = 360;   // let end-pose breathe before the next clip
    const SIGN_BLEND_CLEANUP_MS = Math.round(CROSSFADE_DURATION * 1000) + 20;
    const IDLE_SPEED         = 1.0;
    const ROOT_MOTION_NAMES  = [
        "armature", "root", "hips", "pelvis", "mixamorighips", "bip001", "bip01"
    ];

    // Drag state
    let _isDragging   = false;
    let _dragStartX   = 0;
    let _dragStartRot = 0;

    // ── Library paths ─────────────────────────────────────────────────────────
    const THREE_JS      = "libs/three.min.js";
    const FFLATE_JS     = "libs/fflate.min.js";
    const FBXLOADER_JS  = "libs/FBXLoader.js";
    const GLTFLOADER_JS = "libs/GLTFLoader.js";

    // ── Colour map — fallback when FBX has no embedded textures ──────────────
    const COLOUR_MAP = [
        { keywords: ["eren","yeager","character","body","mesh","s4"], color: 0xC8A882 },
        { keywords: ["face","skin","nail","finger","head","hand","neck","ear"], color: 0xE8B89A },
        { keywords: ["hair"],                                                    color: 0x1A0A00 },
        { keywords: ["cardigan","coat","jacket","outer","blazer"],               color: 0x2C2C2C },
        { keywords: ["topsl","tops","shirt","inner","lace","tee"],               color: 0xD4C9B0 },
        { keywords: ["bottom","pant","trouser","jean"],                          color: 0x1A1A1A },
        { keywords: ["boot","shoe","gum","sole"],                                color: 0x2A2018 },
        { keywords: ["metal","buckle","belt"],                                   color: 0x888888 },
    ];

    // ── Script loader ─────────────────────────────────────────────────────────
    function _loadScript(src) {
        return new Promise((resolve) => {
            if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
            const s = document.createElement("script");
            s.src = src;
            s.onload  = resolve;
            s.onerror = () => { console.warn("Avatar: failed to load:", src); resolve(); };
            document.head.appendChild(s);
        });
    }

    async function _loadDeps() {
        await _loadScript(THREE_JS);
        await _loadScript(FFLATE_JS);
        await _loadScript(FBXLOADER_JS);
        await _loadScript(GLTFLOADER_JS);
        console.log(
            "SAMVAAD Avatar deps — Three.js:", typeof THREE !== "undefined",
            "| FBXLoader:", typeof THREE?.FBXLoader !== "undefined",
            "| GLTFLoader:", typeof THREE?.GLTFLoader !== "undefined"
        );
    }

    function _setAvatarVisible(visible) {
        if (_renderer?.domElement) {
            _renderer.domElement.style.visibility = visible ? "visible" : "hidden";
        }
    }

    function _revealAvatarIfReady() {
        if (!_modelReady || !_assetsReady) return;
        _setAvatarVisible(true);
        if (_loadingEl) _loadingEl.style.display = "none";
        if (_container) {
            _container.querySelectorAll(
                "#avatar-fallback, #sign-avatar-fallback, #avatar"
            ).forEach(el => { el.style.display = "none"; });
        }
    }

    function _createBaseAssetManager() {
        if (typeof THREE?.LoadingManager === "undefined") return;
        _assetsReady = false;
        _baseAssetManager = new THREE.LoadingManager();
        _baseAssetManager.onStart = () => {
            _assetsReady = false;
            if (_loadingEl) _loadingEl.style.display = "flex";
            _setAvatarVisible(false);
        };
        _baseAssetManager.onLoad = () => {
            _assetsReady = true;
            _revealAvatarIfReady();
        };
        _baseAssetManager.onError = (url) => {
            console.warn("Avatar: asset failed while loading:", url);
        };
    }

    // ── Scene setup ───────────────────────────────────────────────────────────
    function _initScene(container) {
        _container = container;
        const w = container.clientWidth  || 500;
        const h = container.clientHeight || 500;

        _renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        _renderer.setSize(w, h);
        _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        _renderer.outputEncoding          = THREE.sRGBEncoding;
        _renderer.physicallyCorrectLights = true;
        _renderer.setClearColor(0x000000, 0);
        _renderer.domElement.style.visibility = "hidden";
        container.appendChild(_renderer.domElement);

        _scene  = new THREE.Scene();
        _camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 10000);

        _scene.add(new THREE.AmbientLight(0xffffff, 2.0));
        const key = new THREE.DirectionalLight(0xffffff, 2.5);
        key.position.set(200, 500, 300); _scene.add(key);
        const fill = new THREE.DirectionalLight(0xffffff, 1.5);
        fill.position.set(-300, 200, 200); _scene.add(fill);
        const front = new THREE.DirectionalLight(0xffffff, 1.2);
        front.position.set(0, 100, 500); _scene.add(front);

        _clock = new THREE.Clock();

        new ResizeObserver(() => {
            const nw = container.clientWidth, nh = container.clientHeight;
            if (!nw || !nh) return;
            _camera.aspect = nw / nh;
            _camera.updateProjectionMatrix();
            _renderer.setSize(nw, nh);
        }).observe(container);

        _setupDragRotation(container);
    }

    // ── Render loop ───────────────────────────────────────────────────────────
    function _renderLoop() {
        requestAnimationFrame(_renderLoop);
        const delta = _clock.getDelta();
        if (_mixer) _mixer.update(delta);
        if (_renderer && _scene && _camera) _renderer.render(_scene, _camera);
    }

    // ── Camera framing ────────────────────────────────────────────────────────
    function _fitCameraHeadToKnees(obj) {
        obj.updateMatrixWorld(true);
        const box    = new THREE.Box3().setFromObject(obj);
        const size   = new THREE.Vector3();
        const center = new THREE.Vector3();
        box.getSize(size); box.getCenter(center);

        const fullHeight  = size.y || 170;
        const thighY      = box.min.y + fullHeight * 0.55;
        const topY        = box.max.y;
        const frameHeight = topY - thighY;
        const midY        = thighY + frameHeight * 0.50;
        const fovRad      = (_camera.fov * Math.PI) / 180;
        const dist        = (frameHeight / 2 / Math.tan(fovRad / 2)) * 1.4;
        const camY        = midY + frameHeight * 0.18;

        _camera.position.set(center.x, camY, center.z + dist);
        _camera.lookAt(center.x, midY, center.z);
        _camera.near = dist * 0.01;
        _camera.far  = dist * 20;
        _camera.updateProjectionMatrix();
    }

    // ── Material handling ─────────────────────────────────────────────────────
    function _applyColours(obj) {
        obj.traverse(child => {
            if (!child.isMesh) return;
            const mats       = Array.isArray(child.material) ? child.material : [child.material];
            const hasTexture = mats.some(m => m && m.map);
            if (hasTexture) {
                mats.forEach(m => {
                    if (!m) return;
                    if (m.map) { m.map.encoding = THREE.sRGBEncoding; m.map.needsUpdate = true; }
                    if (m.shininess !== undefined) m.shininess = 25;
                    if (m.specular) m.specular.setRGB(0.05, 0.05, 0.05);
                    m.needsUpdate = true;
                });
            } else {
                const name = (child.name || "").toLowerCase();
                let colour = 0xC8A882;
                for (const entry of COLOUR_MAP) {
                    if (entry.keywords.some(kw => name.includes(kw))) { colour = entry.color; break; }
                }
                child.material = new THREE.MeshPhongMaterial({
                    color: colour, shininess: 25, specular: new THREE.Color(0x111111)
                });
            }
            child.castShadow = child.receiveShadow = true;
        });
    }

    // ── Pick best clip ────────────────────────────────────────────────────────
    function _pickBestClip(animations) {
        if (!animations || animations.length === 0) return null;
        animations.forEach(clip => {
            if (clip.duration <= 0.01 && clip.tracks.length > 0) {
                let maxTime = 0;
                clip.tracks.forEach(track => {
                    if (track.times && track.times.length > 0)
                        maxTime = Math.max(maxTime, track.times[track.times.length - 1]);
                });
                if (maxTime > 0) {
                    clip.duration = maxTime;
                    console.log(`Avatar: repaired clip "${clip.name}" → ${maxTime.toFixed(2)}s`);
                }
            }
        });
        const bestClip = animations.reduce((best, clip) =>
            clip.duration > best.duration ? clip : best, animations[0]);
        return _stabilizeClip(bestClip);
    }

    function _isRootMotionTrack(trackName) {
        if (!trackName) return false;
        const parts = String(trackName).split(".");
        if (parts.length < 2) return false;

        const nodeName = parts[0].toLowerCase().replace(/[^a-z0-9]/g, "");
        const propName = parts[parts.length - 1].toLowerCase();
        if (propName !== "position") return false;

        return ROOT_MOTION_NAMES.some(name => nodeName === name || nodeName.includes(name));
    }

    function _stabilizeClip(clip) {
        if (!clip?.tracks?.length || typeof THREE?.AnimationClip === "undefined") {
            return clip;
        }

        const filteredTracks = clip.tracks.filter(track => !_isRootMotionTrack(track.name));
        if (filteredTracks.length === clip.tracks.length) {
            return clip;
        }

        console.log(
            `Avatar: stabilized clip "${clip.name}" by removing ${clip.tracks.length - filteredTracks.length} root-motion track(s)`
        );
        return new THREE.AnimationClip(clip.name, clip.duration, filteredTracks);
    }

    // ── Raw loaders ───────────────────────────────────────────────────────────
    function _loadFBX(path, onLoaded, manager = null) {
        if (typeof THREE?.FBXLoader === "undefined") { onLoaded(null, []); return; }
        const loader = new THREE.FBXLoader(manager || undefined);
        loader.setResourcePath(window.location.origin + "/" + _animFolder);
        loader.load(path,
            obj  => onLoaded(obj, obj.animations || []),
            null,
            err  => { console.warn("Avatar: FBX failed:", path, err?.message || err); onLoaded(null, []); }
        );
    }

    function _loadGLB(path, onLoaded, manager = null) {
        if (typeof THREE?.GLTFLoader === "undefined") { onLoaded(null, []); return; }
        const loader = new THREE.GLTFLoader(manager || undefined);
        loader.load(path,
            gltf => onLoaded(gltf.scene, gltf.animations || []),
            null,
            err  => { console.warn("Avatar: GLB failed:", path, err?.message || err); onLoaded(null, []); }
        );
    }

    // ── Try FBX first, fall back to GLB ──────────────────────────────────────
    function _tryLoad(base, onLoaded, manager = null) {
        const fbxPath = _animFolder + base + ".fbx";
        const glbPath = _animFolder + base + ".glb";
        fetch(fbxPath, { method: "HEAD" })
            .then(r => r.ok ? _loadFBX(fbxPath, onLoaded, manager) : _loadGLB(glbPath, onLoaded, manager))
            .catch(()  => _loadGLB(glbPath, onLoaded, manager));
    }

    // ── Check if animation file exists (cached) ───────────────────────────────
    function _fileExists(base) {
        const fbx = _animFolder + base + ".fbx";
        const glb = _animFolder + base + ".glb";
        if (fbx in _existsCache) return Promise.resolve(_existsCache[fbx] || (_existsCache[glb] || false));
        return fetch(fbx, { method: "HEAD" })
            .then(r => {
                _existsCache[fbx] = r.ok;
                if (r.ok) return true;
                return fetch(glb, { method: "HEAD" }).then(r2 => { _existsCache[glb] = r2.ok; return r2.ok; });
            })
            .catch(() => { _existsCache[fbx] = false; return false; });
    }

    // ── Load base model once ──────────────────────────────────────────────────
    function _loadBaseModel(base, onReady, onFail) {
        if (_modelLoading || _modelReady) return;
        _modelLoading = true;

        _tryLoad(base, (obj, animations) => {
            if (!obj) {
                console.warn("Avatar: base model load failed for:", base);
                _modelLoading = false;
                if (_loadingEl) _loadingEl.style.display = "none";
                if (onFail) onFail();
                return;
            }

            _applyColours(obj);
            obj.rotation.y = _rotationY;
            _scene.add(obj);

            _model        = obj;
            _mixer        = new THREE.AnimationMixer(obj);
            _modelReady   = true;
            _modelLoading = false;

            _fitCameraHeadToKnees(obj);

            const clip = _pickBestClip(animations);
            if (clip) _clipCache[base] = clip;

            console.log(`Avatar: model ready (source: "${base}") ✓`);
            _revealAvatarIfReady();
            onReady();
        }, _baseAssetManager);
    }

    // ── Fetch clip for a key, using cache ─────────────────────────────────────
    function _fetchClip(base, onClip) {
        if (base in _clipCache) { onClip(_clipCache[base]); return; }
        _tryLoad(base, (_obj, animations) => {
            const clip = _pickBestClip(animations);
            _clipCache[base] = clip || null;
            onClip(clip);
        });
    }

    function _getClipDurationMs(base, fallbackMs = 3000) {
        return new Promise(resolve => {
            _fetchClip(base, clip => {
                const seconds = clip?.duration || 0;
                if (seconds > 0.01) {
                    resolve(Math.round((seconds * 1000) / Math.max(_speed, 0.1)));
                    return;
                }
                resolve(fallbackMs);
            });
        });
    }

    // ── Crossfade to a clip ───────────────────────────────────────────────────
    function _crossfadeToClip(clip, label, onDone) {
        if (_onSignChange) _onSignChange(label || "");

        if (!clip || !_mixer || !_model) {
            setTimeout(onDone, 16);
            return;
        }

        const incoming = _mixer.clipAction(clip);
        incoming.reset();
        incoming.enabled           = true;
        incoming.setLoop(THREE.LoopOnce, 1);
        incoming.clampWhenFinished = true;
        incoming.timeScale         = _speed;

        const prev = _currentAction;
        const prevIsIdle = (prev === _idleAction);

        if (prev && prev !== incoming) {
            if (prevIsIdle) {
                // Idle is LoopRepeat — we must crossFadeTo (not From) so we
                // control the direction, then hard-stop idle after the blend
                // window so it cannot keep influencing the mixer.
                prev.crossFadeTo(incoming, CROSSFADE_DURATION, false);
                setTimeout(() => {
                    if (_idleAction) {
                        _idleAction.stop();
                        _idleAction.enabled = false;
                        _idleAction = null;
                    }
                }, SIGN_BLEND_CLEANUP_MS);
            } else {
                // Previous was a sign (LoopOnce, clamped). Blend at fixed speed
                // so time-warping does not make the gesture feel cut or frozen.
                incoming.crossFadeFrom(prev, CROSSFADE_DURATION, false);
                setTimeout(() => {
                    if (prev !== incoming) {
                        prev.stop();
                        prev.enabled = false;
                    }
                }, SIGN_BLEND_CLEANUP_MS);
            }
        }

        incoming.play();
        _currentAction = incoming;
        if (_model) _model.rotation.y = _rotationY;

        let done = false;
        function finish() {
            if (done) return;
            done = true;
            _mixer.removeEventListener("finished", onFinished);
            if (_model) _model.rotation.y = _rotationY;
            setTimeout(() => {
                requestAnimationFrame(() => onDone());
            }, CLIP_SETTLE_MS);
        }
        function onFinished(e) { if (e.action === incoming) finish(); }
        _mixer.addEventListener("finished", onFinished);

        const durMs = (clip.duration * 1000) / Math.max(_speed, 0.1);
        setTimeout(finish, Math.max(durMs, MIN_SIGN_MS));
    }

    function _resolveSingleSignKey(rawKey) {
        if (!rawKey) return null;
        const key = String(rawKey).trim();
        if (!key) return null;

        const lower = key.toLowerCase();
        if (/^[a-z]$/.test(lower)) return lower.toUpperCase();
        return lower;
    }

    // ── Play idle loop ────────────────────────────────────────────────────────
    function _playIdle(onDone) {
        _fetchClip(T.IDLE, clip => {
            if (!clip || !_mixer || !_model) {
                if (onDone) onDone();
                return;
            }

            if (_onSignChange) _onSignChange("");

            const action = _mixer.clipAction(clip);
            // Full reset so idle always starts from frame 0 cleanly,
            // especially important after being stopped by _crossfadeToClip.
            action.stop();
            action.reset();
            action.enabled           = true;
            action.setLoop(THREE.LoopRepeat, Infinity);
            action.clampWhenFinished = false;
            action.timeScale         = IDLE_SPEED;

            // Smooth crossfade from the last sign back into idle.
            // The previous sign is clamped at its last frame, so blending
            // from it into idle looks natural.
            if (_currentAction && _currentAction !== action) {
                action.crossFadeFrom(_currentAction, CROSSFADE_DURATION, true);
            }

            action.play();
            _currentAction = action;
            _idleAction    = action;
            if (_model) _model.rotation.y = _rotationY;
            if (onDone) onDone();
        });
    }

    // ── Determine if a sign key uses the left hand ────────────────────────────
    function _isLeftHand(key) {
        return LEFT_HAND_KEYS.has(key.toLowerCase());
    }

    // ── Build queue with SMART raise/lower logic ──────────────────────────────
    //
    // Rules:
    //  • Each word is first checked for a whole-word FBX file.
    //  • Words WITH their own FBX are "whole-word" items — played directly,
    //    NO raise/lower around them (they typically start from a neutral pose).
    //  • Words WITHOUT an FBX are split into individual letters
    //    ("fingerspelled run"). Raise hand before the run, lower after the run.
    //  • Adjacent fingerspelled letters that use the SAME hand are merged into
    //    one run (one raise, one lower).
    //  • When a fingerspelled run ends and a whole-word sign follows (or vice
    //    versa), the hand is lowered first.
    //  • A SPACE marker is inserted between every word (gracefully skipped at
    //    playback time if space.fbx / space.glb does not exist yet).
    //
    async function _buildQueue(words) {

        // Step 1 — resolve each word to either a whole-word key or a letter array
        // type: "word" | "letters"
        // hand: "right" | "left" (only meaningful for "letters" runs)
        const resolved = [];

        for (let wi = 0; wi < words.length; wi++) {
            const word  = words[wi];
            const lower = word.toLowerCase();
            const exists = await _fileExists(lower);

            if (exists) {
                const key  = lower.length === 1 ? lower.toUpperCase() : lower;
                const left = _isLeftHand(lower);
                resolved.push({ type: "word", key, left });
            } else {
                // Fingerspell each alphabetic character
                const letters = [];
                for (const ch of lower) {
                    if (/[a-z]/.test(ch)) letters.push(ch.toUpperCase());
                }
                if (letters.length > 0) {
                    // All letters are right-hand (left-hand signs like YES/NO
                    // always have their own whole-word FBX)
                    resolved.push({ type: "letters", letters, left: false });
                }
            }

            // Insert a SPACE marker between words (not after the last word)
            if (wi < words.length - 1) {
                resolved.push({ type: "space" });
            }
        }

        if (resolved.length === 0) return [];

        // Step 2 — build the final flat queue with raise/lower transitions
        const queue = [];
        let activeHand = null; // "right" | "left" | null

        for (const item of resolved) {

            if (item.type === "space") {
                // Space goes in as-is; if activeHand is raised we DON'T lower yet
                // (the space gesture is performed with whatever hand is active)
                queue.push(T.SPACE);
                continue;
            }

            if (item.type === "word") {
                // Whole-word animation — lower any raised hand first, then play
                if (activeHand === "right") { queue.push(T.LOWER_RIGHT); activeHand = null; }
                if (activeHand === "left")  { queue.push(T.LOWER_LEFT);  activeHand = null; }
                queue.push(item.key);
                // Hand returns to neutral after a whole-word sign
                activeHand = null;
                continue;
            }

            if (item.type === "letters") {
                const thisHand = item.left ? "left" : "right";

                if (activeHand !== thisHand) {
                    // Lower previous hand if different
                    if (activeHand === "right") queue.push(T.LOWER_RIGHT);
                    if (activeHand === "left")  queue.push(T.LOWER_LEFT);
                    // Raise new hand
                    if (thisHand === "right") queue.push(T.RAISE_RIGHT);
                    if (thisHand === "left")  queue.push(T.RAISE_LEFT);
                    activeHand = thisHand;
                }
                // Push all letters (hand stays raised between them)
                for (const letter of item.letters) queue.push(letter);
                continue;
            }
        }

        // Lower hand at the very end if still raised
        if (activeHand === "right") queue.push(T.LOWER_RIGHT);
        if (activeHand === "left")  queue.push(T.LOWER_LEFT);

        return queue;
    }

    // ── Queue player ──────────────────────────────────────────────────────────
    function _playNext(token = _playbackToken) {
        if (token !== _playbackToken) return;

        if (_queue.length === 0) {
            _isPlaying = false;
            if (_onSignChange) _onSignChange("");
            _playIdle();
            return;
        }

        _isPlaying    = true;
        const signKey = _queue.shift();
        const isTransition = TRANSITION_KEYS.has(signKey);

        _fetchClip(signKey, clip => {
            if (token !== _playbackToken) return;
            _crossfadeToClip(clip, isTransition ? null : signKey, () => _playNext(token));
        });
    }

    // ── Drag-to-rotate ────────────────────────────────────────────────────────
    function _setupDragRotation(container) {
        container.addEventListener("mousedown", e => {
            _isDragging = true; _dragStartX = e.clientX; _dragStartRot = _rotationY;
            container.style.cursor = "grabbing";
        });
        window.addEventListener("mousemove", e => {
            if (!_isDragging) return;
            _rotationY = _dragStartRot + (e.clientX - _dragStartX) * 0.01;
            if (_model) _model.rotation.y = _rotationY;
        });
        window.addEventListener("mouseup", () => { _isDragging = false; container.style.cursor = "grab"; });
        container.addEventListener("touchstart", e => {
            _isDragging = true; _dragStartX = e.touches[0].clientX; _dragStartRot = _rotationY;
        }, { passive: true });
        window.addEventListener("touchmove", e => {
            if (!_isDragging) return;
            _rotationY = _dragStartRot + (e.touches[0].clientX - _dragStartX) * 0.01;
            if (_model) _model.rotation.y = _rotationY;
        }, { passive: true });
        window.addEventListener("touchend", () => { _isDragging = false; });
        container.style.cursor = "grab";
    }

    // ── Wait until model is ready, then call fn ───────────────────────────────
    function _whenReady(fn) {
        if (_modelReady) { fn(); return; }
        const poll = setInterval(() => { if (_modelReady) { clearInterval(poll); fn(); } }, 100);
    }

    // =========================================================================
    //  PUBLIC API
    // =========================================================================

    /**
     * Initialise the avatar.
     */
    api.init = function ({
        containerId,
        loadingId,
        animationsFolder,
        baseAnimation = "A",
        onSignChange  = null,
    }) {
        // Returns a Promise that resolves ONLY after the model is fully loaded
        // and idle is playing. This lets index.html do:
        //   await SAMVAADAvatar.init({...});
        //   SAMVAADAvatar.playIntro([...]);   // guaranteed model is ready
        return new Promise(async (resolve) => {
            if (_initialized) { resolve(); return; }
            _initialized  = true;
            _onSignChange = onSignChange;
            if (animationsFolder) _animFolder = animationsFolder;

            const container = document.getElementById(containerId);
            if (!container) { console.error("Avatar: container not found:", containerId); resolve(); return; }
            if (loadingId) _loadingEl = document.getElementById(loadingId);

            try {
                if (_loadingEl) _loadingEl.style.display = "flex";
                await _loadDeps();
                _createBaseAssetManager();
                _initScene(container);
                _renderLoop();

                function onModelReady() {
                    _playIdle();
                    console.log("SAMVAAD Avatar ready ✓");
                    resolve();   // ← Promise resolves HERE, after model + idle
                }

                // Try idle.fbx first (neutral pose), fall back to baseAnimation
                _loadBaseModel("idle", onModelReady, () => {
                    _loadBaseModel(baseAnimation, onModelReady, () => {
                        console.error("Avatar: all model loads failed");
                        resolve();
                    });
                });
            } catch (err) {
                console.error("Avatar init failed:", err);
                if (_loadingEl) _loadingEl.style.display = "none";
                resolve();
            }
        });
    };

    /**
     * Play an intro sequence then loop (index.html).
     *
     * Loop:  idle (2 s)  →  sequence  →  idle (3 s)  →  sequence  →  …
     *
     * @param {string[]} sequence — e.g. ["hello", "welcome", "to", "samvaad"]
     */
    api.playIntro = function (sequence) {
        if (!sequence || sequence.length === 0) return;

        function runKeys(keys, onAllDone) {
            if (keys.length === 0) { onAllDone(); return; }
            const key = keys.shift();
            const isTrans = TRANSITION_KEYS.has(key);
            _fetchClip(key, clip => {
                _crossfadeToClip(clip, isTrans ? null : key, () => runKeys(keys, onAllDone));
            });
        }

        async function start() {
            console.log("Avatar intro: building queue...");
            const builtQueue = await _buildQueue(sequence);
            console.log("Avatar intro: queue ready:", builtQueue);

            if (!builtQueue || builtQueue.length === 0) {
                console.warn("Avatar intro: empty queue, retry 2 s");
                setTimeout(start, 2000);
                return;
            }

            const fullIdleMs = await _getClipDurationMs(T.IDLE, 3000);

            function cycle() {
                console.log("Avatar intro: short idle", IDLE_INTRO_MS, "ms");
                _playIdle();
                setTimeout(() => {
                    runKeys([...builtQueue], () => {
                        console.log("Avatar intro: full idle", fullIdleMs, "ms");
                        _playIdle();
                        setTimeout(cycle, fullIdleMs);
                    });
                }, IDLE_INTRO_MS);
            }

            cycle();
        }

        _whenReady(start);
    };

    /**
     * Translate text to sign animations (sign.html).
     * Automatically checks server for whole-word files before fingerspelling.
     * Wraps fingerspelled runs with raise/lower hand transitions.
     * Whole-word signs are played directly without raise/lower.
     * Space gesture inserted between words.
     * Returns to idle when done.
     *
     * @param {string} text — the text to sign
     */
    api.convertToSign = async function (text) {
        if (!text || !text.trim()) return;

        _playbackToken += 1;
        _queue     = [];
        _isPlaying = false;
        const token = _playbackToken;

        const words = text.trim().toLowerCase().split(/\s+/).filter(Boolean);
        const queue = await _buildQueue(words);

        if (token !== _playbackToken || queue.length === 0) return;
        _queue = queue;

        _whenReady(() => _playNext(token));
    };

    /**
     * Play a single sign cleanly for learn mode without word-spacing
     * or raise/lower transition choreography.
     *
     * @param {string} key - the exact sign key to play
     */
    api.playLearnSign = function (key) {
        const resolvedKey = _resolveSingleSignKey(key);
        if (!resolvedKey) return;

        _playbackToken += 1;
        const token = _playbackToken;
        _queue = [resolvedKey];
        _isPlaying = false;

        _whenReady(() => _playNext(token));
    };

    /**
     * Stop all playback and return to idle.
     */
    api.stop = function () {
        _playbackToken += 1;
        _queue     = [];
        _isPlaying = false;
        if (_onSignChange) _onSignChange("");
        _whenReady(() => _playIdle());
    };

    /**
     * Change playback speed.
     * @param {number} speed — 0.5 / 1.0 / 2.0
     */
    api.setSpeed = function (speed) {
        _speed = speed;
        if (_currentAction && _currentAction !== _idleAction) {
            _currentAction.timeScale = speed;
        }
        if (_idleAction) {
            _idleAction.timeScale = IDLE_SPEED;
        }
    };

    /**
     * Register or update the sign-change callback after init.
     */
    api.onSignChange = function (fn) { _onSignChange = fn; };

    /** @returns {boolean} true once model is loaded and renderer is running */
    api.isReady = function () { return _initialized && _modelReady; };

    return api;

})();

