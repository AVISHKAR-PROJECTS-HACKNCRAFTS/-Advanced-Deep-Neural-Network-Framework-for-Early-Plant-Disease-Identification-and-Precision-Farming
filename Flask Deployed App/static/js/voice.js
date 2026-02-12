// ── Voice & Translation Module ───────────────────────────────────────────
// Provides: speech recognition (voice input for disease/fertilizer search),
// speech synthesis (reads results aloud), and page translation.

(function () {
    'use strict';

    // ── Language Config ──────────────────────────────────────────────────
    var LANGUAGES = {
        en: { speech: 'en-US',  name: 'English' },
        hi: { speech: 'hi-IN',  name: 'Hindi' },
        ta: { speech: 'ta-IN',  name: 'Tamil' },
        te: { speech: 'te-IN',  name: 'Telugu' },
        bn: { speech: 'bn-IN',  name: 'Bengali' },
        mr: { speech: 'mr-IN',  name: 'Marathi' },
        gu: { speech: 'gu-IN',  name: 'Gujarati' },
        kn: { speech: 'kn-IN',  name: 'Kannada' },
        ml: { speech: 'ml-IN',  name: 'Malayalam' },
        pa: { speech: 'pa-IN',  name: 'Punjabi' }
    };

    var currentLang = 'en';
    var isListening = false;
    var isSpeaking = false;
    var recognition = null;
    var synth = window.speechSynthesis;

    // ── Speech Recognition (Input) ──────────────────────────────────────

    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    function initRecognition() {
        if (!SpeechRecognition) return null;

        var rec = new SpeechRecognition();
        rec.continuous = false;
        rec.interimResults = false;
        rec.lang = LANGUAGES[currentLang].speech;

        rec.onresult = function (e) {
            var transcript = e.results[0][0].transcript.toLowerCase().trim();
            handleVoiceCommand(transcript);
        };

        rec.onend = function () {
            isListening = false;
            updateMicButton(false);
        };

        rec.onerror = function (e) {
            console.warn('Speech recognition error:', e.error);
            isListening = false;
            updateMicButton(false);
            showFeedback('Could not recognize speech. Please try again.', 'error');
        };

        return rec;
    }

    function handleVoiceCommand(text) {
        // Navigation commands
        var routes = {
            'home': '/', 'index': '/', 'go home': '/',
            'market': '/market', 'supplement': '/market', 'shop': '/market',
            'alert': '/alerts', 'alerts': '/alerts', 'community': '/alerts'
        };

        for (var key in routes) {
            if (text.indexOf(key) !== -1 && text.length < 30) {
                window.location.href = routes[key];
                return;
            }
        }

        // Disease / fertilizer search — user said a plant or disease name
        showFeedback('Searching for: "' + text + '"...', 'searching');
        searchDisease(text);
    }

    function searchDisease(query) {
        fetch('/api/search-disease?q=' + encodeURIComponent(query))
            .then(function (res) { return res.json(); })
            .then(function (data) {
                if (data.results && data.results.length > 0) {
                    showVoiceResults(data.results, query);
                } else {
                    showFeedback('No match found for "' + query + '". Try saying a plant or disease name like "tomato" or "apple scab".', 'error');
                }
            })
            .catch(function () {
                showFeedback('Search failed. Please try again.', 'error');
            });
    }

    function showVoiceResults(results, query) {
        // Remove any existing overlay
        var existing = document.getElementById('voice-results-overlay');
        if (existing) existing.remove();

        var overlay = document.createElement('div');
        overlay.id = 'voice-results-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(5px);';

        var panel = document.createElement('div');
        panel.style.cssText = 'background:#fff;border-radius:24px;max-width:600px;width:100%;max-height:80vh;overflow-y:auto;padding:30px;box-shadow:0 25px 80px rgba(0,0,0,0.3);';

        // Header
        var header = document.createElement('div');
        header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;';

        var title = document.createElement('h4');
        title.style.cssText = 'margin:0;color:#1a5c2e;font-weight:700;';
        title.textContent = 'Results for "' + query + '"';
        header.appendChild(title);

        var closeBtn = document.createElement('button');
        closeBtn.style.cssText = 'background:none;border:none;font-size:24px;cursor:pointer;color:#999;padding:5px 10px;';
        closeBtn.textContent = '\u00d7';
        closeBtn.onclick = function () { overlay.remove(); };
        header.appendChild(closeBtn);
        panel.appendChild(header);

        // Result cards
        results.forEach(function (r) {
            var card = document.createElement('div');
            card.style.cssText = 'background:#f8fff8;border-radius:16px;padding:20px;margin-bottom:15px;border-left:4px solid ' + (r.is_healthy ? '#4ECC5A' : '#ff6b6b') + ';';

            var nameRow = document.createElement('div');
            nameRow.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;';

            var diseaseName = document.createElement('h5');
            diseaseName.style.cssText = 'margin:0;color:#1a5c2e;font-weight:700;font-size:1.1rem;';
            diseaseName.textContent = r.disease_name;
            nameRow.appendChild(diseaseName);

            var sevBadge = document.createElement('span');
            var sevColor = { Critical: '#c62828', Moderate: '#f9a825', Mild: '#ef6c00', Healthy: '#2d8f4e' };
            var sevBg = { Critical: '#ffebee', Moderate: '#fff8e1', Mild: '#fff3e0', Healthy: '#e8f5e9' };
            sevBadge.style.cssText = 'padding:4px 12px;border-radius:50px;font-size:0.75rem;font-weight:700;background:' + (sevBg[r.severity] || '#f5f5f5') + ';color:' + (sevColor[r.severity] || '#666') + ';';
            sevBadge.textContent = r.severity;
            nameRow.appendChild(sevBadge);
            card.appendChild(nameRow);

            // Supplement/Fertilizer info
            var suppLabel = document.createElement('p');
            suppLabel.style.cssText = 'font-size:0.8rem;color:#888;margin:0 0 5px 0;text-transform:uppercase;letter-spacing:0.5px;';
            suppLabel.textContent = r.is_healthy ? 'Recommended Fertilizer' : 'Recommended Supplement';
            card.appendChild(suppLabel);

            var suppName = document.createElement('p');
            suppName.style.cssText = 'font-size:1rem;font-weight:600;color:#333;margin:0 0 12px 0;';
            suppName.textContent = r.supplement_name;
            card.appendChild(suppName);

            // Action buttons row
            var actions = document.createElement('div');
            actions.style.cssText = 'display:flex;gap:10px;flex-wrap:wrap;';

            // Speak button
            var speakBtn = document.createElement('button');
            speakBtn.style.cssText = 'background:linear-gradient(135deg,#4ECC5A,#2d8f4e);color:#fff;border:none;padding:8px 20px;border-radius:50px;font-weight:600;cursor:pointer;font-size:0.85rem;display:inline-flex;align-items:center;gap:6px;';
            var speakIcon = document.createElement('i');
            speakIcon.className = 'fas fa-volume-up';
            speakBtn.appendChild(speakIcon);
            speakBtn.appendChild(document.createTextNode(' Listen'));
            speakBtn.onclick = (function (disease) {
                return function () {
                    var label = disease.is_healthy ? 'fertilizer' : 'supplement';
                    var speechText = disease.disease_name + '. Severity: ' + disease.severity +
                        '. Recommended ' + label + ': ' + disease.supplement_name;
                    speakText(speechText);
                };
            })(r);
            actions.appendChild(speakBtn);

            // Buy link
            if (r.buy_link) {
                var buyBtn = document.createElement('a');
                buyBtn.href = r.buy_link;
                buyBtn.target = '_blank';
                buyBtn.rel = 'noopener';
                buyBtn.style.cssText = 'background:linear-gradient(135deg,#5f27cd,#341f97);color:#fff;border:none;padding:8px 20px;border-radius:50px;font-weight:600;cursor:pointer;font-size:0.85rem;text-decoration:none;display:inline-flex;align-items:center;gap:6px;';
                var cartIcon = document.createElement('i');
                cartIcon.className = 'fas fa-shopping-cart';
                buyBtn.appendChild(cartIcon);
                buyBtn.appendChild(document.createTextNode(' Buy'));
                actions.appendChild(buyBtn);
            }

            card.appendChild(actions);
            panel.appendChild(card);
        });

        overlay.appendChild(panel);

        // Close on overlay background click
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay) overlay.remove();
        });

        document.body.appendChild(overlay);

        // Auto-speak the first result
        var first = results[0];
        var label = first.is_healthy ? 'fertilizer' : 'supplement';
        var speechText = 'Found ' + results.length + ' result' + (results.length > 1 ? 's' : '') + '. ' +
            first.disease_name + '. Severity: ' + first.severity +
            '. Recommended ' + label + ': ' + first.supplement_name;
        speakText(speechText);
    }

    function showFeedback(message, type) {
        var feedback = document.getElementById('voice-feedback');
        if (!feedback) return;

        feedback.textContent = message;
        feedback.style.display = 'block';

        if (type === 'error') {
            feedback.style.background = 'rgba(255, 71, 87, 0.1)';
            feedback.style.color = '#ff4757';
        } else if (type === 'searching') {
            feedback.style.background = 'rgba(78, 204, 90, 0.1)';
            feedback.style.color = '#2d8f4e';
        }

        if (type !== 'searching') {
            setTimeout(function () { feedback.style.display = 'none'; }, 5000);
        }
    }

    function updateMicButton(active) {
        var btn = document.getElementById('mic-btn');
        if (!btn) return;
        if (active) {
            btn.classList.add('mic-active');
        } else {
            btn.classList.remove('mic-active');
        }
    }

    // Global: toggle microphone
    window.toggleMic = function () {
        if (!SpeechRecognition) {
            alert('Speech recognition is not supported in this browser. Please use Chrome.');
            return;
        }

        if (isListening) {
            recognition.stop();
            isListening = false;
            updateMicButton(false);
            return;
        }

        recognition = initRecognition();
        if (!recognition) return;

        recognition.lang = LANGUAGES[currentLang].speech;
        recognition.start();
        isListening = true;
        updateMicButton(true);
        showFeedback('Listening... Say a plant or disease name (e.g. "tomato", "apple scab")', 'searching');
    };

    // ── Speech Synthesis (Output) ───────────────────────────────────────

    // Chrome workaround: keep speech alive (Chrome pauses after ~15s)
    var resumeTimer = null;

    // Ensure voices are loaded (Chrome loads them asynchronously)
    var voicesReady = false;
    function loadVoices() {
        var v = synth.getVoices();
        if (v.length > 0) voicesReady = true;
    }
    loadVoices();
    if (synth.onvoiceschanged !== undefined) {
        synth.onvoiceschanged = loadVoices;
    }

    function doSpeak(text) {
        var utterance = new SpeechSynthesisUtterance(text.trim());
        utterance.lang = LANGUAGES[currentLang].speech;
        utterance.rate = 0.9;

        // Try to pick a matching voice explicitly
        var voices = synth.getVoices();
        var langCode = LANGUAGES[currentLang].speech;
        for (var i = 0; i < voices.length; i++) {
            if (voices[i].lang === langCode) {
                utterance.voice = voices[i];
                break;
            }
        }

        utterance.onstart = function () {
            isSpeaking = true;
            updateSpeakButton(true);
        };

        utterance.onend = function () {
            clearInterval(resumeTimer);
            isSpeaking = false;
            updateSpeakButton(false);
        };

        utterance.onerror = function () {
            clearInterval(resumeTimer);
            isSpeaking = false;
            updateSpeakButton(false);
        };

        synth.speak(utterance);
        isSpeaking = true;
        updateSpeakButton(true);

        // Chrome bug fix: periodically call resume() to prevent freezing
        clearInterval(resumeTimer);
        resumeTimer = setInterval(function () {
            if (!synth.speaking) {
                clearInterval(resumeTimer);
            } else {
                synth.pause();
                synth.resume();
            }
        }, 10000);
    }

    function speakText(text) {
        if (!text || !text.trim()) return;

        // Cancel any current speech
        synth.cancel();
        clearInterval(resumeTimer);

        // Chrome fix: small delay after cancel() before speaking again
        setTimeout(function () {
            doSpeak(text);
        }, 100);
    }

    window.toggleSpeech = function () {
        if (isSpeaking) {
            synth.cancel();
            clearInterval(resumeTimer);
            isSpeaking = false;
            updateSpeakButton(false);
            return;
        }

        var el = document.getElementById('speakable-text');
        if (!el) return;

        var text = (el.textContent || el.innerText || '').trim();
        if (!text) return;

        speakText(text);
    };

    function updateSpeakButton(active) {
        var btn = document.getElementById('listen-btn');
        if (!btn) return;
        var icon = btn.querySelector('i');
        if (!icon) return;
        if (active) {
            icon.className = 'fas fa-stop';
            btn.title = 'Stop speaking';
        } else {
            icon.className = 'fas fa-volume-up';
            btn.title = 'Listen to results';
        }
    }

    // ── Translation ─────────────────────────────────────────────────────

    window.translatePage = function (lang) {
        currentLang = lang;

        // Store originals on first call
        var elements = document.querySelectorAll('[data-translatable]');
        elements.forEach(function (el) {
            if (!el.hasAttribute('data-original-text')) {
                el.setAttribute('data-original-text', el.textContent);
            }
        });

        // If English, restore originals immediately
        if (lang === 'en') {
            elements.forEach(function (el) {
                el.textContent = el.getAttribute('data-original-text');
            });
            return;
        }

        // Translate each element
        elements.forEach(function (el) {
            var original = el.getAttribute('data-original-text');
            if (!original || !original.trim()) return;

            fetch('/api/translate?text=' + encodeURIComponent(original) + '&dest=' + lang)
                .then(function (res) { return res.json(); })
                .then(function (data) {
                    if (data.translated) {
                        el.textContent = data.translated;
                    }
                })
                .catch(function (err) {
                    console.warn('Translation failed for element:', err);
                });
        });
    };

    // ── Language Selector Handler ────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', function () {
        var selector = document.getElementById('language-selector');
        if (selector) {
            selector.addEventListener('change', function () {
                window.translatePage(this.value);
            });
        }
    });

})();
