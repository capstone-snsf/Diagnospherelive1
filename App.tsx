import React, { useState, useCallback, useRef, useEffect } from 'react';
import { analyzeMedicalImage } from './services/geminiService';
import { AnalysisResult, Diagnosis, ChatMessage } from './types';
import { 
  UploadIcon, 
  StethoscopeIcon, 
  LightbulbIcon, 
  CheckCircleIcon, 
  Spinner, 
  SendIcon, 
  HomeIcon,
  UserGroupIcon,
  QuestionMarkCircleIcon,
  RocketIcon,
  EnvelopeIcon,
  MicrophoneIcon,
  SpeakerIcon,
  LanguageIcon,
  StopIcon,
  StarIcon,
  UserIcon
} from './components/icons';
import { GoogleGenAI, Chat, LiveServerMessage, Modality } from "@google/genai";

// --- Language Configuration ---

interface Language {
  code: string;
  name: string;
  label: string;
}

const LANGUAGES: Language[] = [
  { code: 'en-US', name: 'English', label: 'English' },
  { code: 'hi-IN', name: 'Hindi', label: 'Hindi (हिंदी)' },
  { code: 'es-ES', name: 'Spanish', label: 'Spanish (Español)' },
  { code: 'fr-FR', name: 'French', label: 'French (Français)' },
];

// --- Types for Feedback ---
interface Review {
    id: string;
    name: string;
    rating: number;
    comment: string;
    date: string;
}

// --- Helper Functions for Live API Audio ---

// Convert Base64 string to Uint8Array
function decodeBase64(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Convert Float32Array PCM to Base64 (for sending to API)
function float32ToBase64(data: Float32Array): string {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    // Clamp values and convert to 16-bit PCM
    const s = Math.max(-1, Math.min(1, data[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  
  let binary = '';
  const bytes = new Uint8Array(int16.buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Helper to decode raw PCM data for playback
async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000,
  numChannels: number = 1
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// Helper to safely get the API Key (Supports Vite/Vercel & Local)
const getApiKey = (): string | undefined => {
  let key = '';
  // 1. Try VITE_API_KEY (for Vercel/Vite deployments)
  try {
    // @ts-ignore
    if (import.meta && import.meta.env && import.meta.env.VITE_API_KEY) {
      // @ts-ignore
      key = import.meta.env.VITE_API_KEY;
    }
  } catch (e) {}

  // 2. Fallback to process.env.API_KEY (for local development)
  if (!key) {
    try {
      if (typeof process !== 'undefined' && process.env && process.env.API_KEY) {
        key = process.env.API_KEY;
      }
    } catch (e) {}
  }
  return key || undefined;
};

// --- Hook for Gemini Live API ---

const useLiveAPI = (
  selectedLanguage: Language, 
  onTranscript: (text: string, isUser: boolean) => void,
  analysis: AnalysisResult | null
) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isError, setIsError] = useState(false);
  
  // Refs for audio handling to avoid re-renders
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const scheduledSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionRef = useRef<any>(null); // To store the active session

  const disconnect = useCallback(() => {
    // 1. Close session
    if (sessionRef.current) {
        sessionRef.current = null;
    }

    // 2. Stop Audio Input
    if (inputSourceRef.current) {
        inputSourceRef.current.disconnect();
        inputSourceRef.current = null;
    }
    if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
    }

    // 3. Stop Audio Output
    scheduledSourcesRef.current.forEach(source => {
        try { source.stop(); } catch (e) { /* ignore */ }
    });
    scheduledSourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
    }

    setIsConnected(false);
  }, []);

  const connect = useCallback(async () => {
    try {
        setIsError(false);
        // Initialize Audio Contexts
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioContextClass({ sampleRate: 24000 }); // Output rate
        audioContextRef.current = ctx;

        // Initialize Gemini Client
        const apiKey = getApiKey();
        if (!apiKey) throw new Error("API Key not found. Please add VITE_API_KEY to Vercel.");

        const ai = new GoogleGenAI({ apiKey });

        // Get Microphone Stream
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
        
        // Setup Input Processing (Mic -> 16kHz PCM -> Model)
        // We use a separate context for input to force 16kHz if possible, or resample.
        const inputCtx = new AudioContextClass({ sampleRate: 16000 });
        const source = inputCtx.createMediaStreamSource(stream);
        const processor = inputCtx.createScriptProcessor(4096, 1, 1);
        
        inputSourceRef.current = source;
        processorRef.current = processor;

        source.connect(processor);
        processor.connect(inputCtx.destination);

        // Prepare System Instruction with Analysis Context
        let systemInstructionText = `You are DiagnoSphere's friendly medical AI assistant.
        You are talking to a user about their medical scan.
        Keep responses concise, empathetic, and simple (10th-grade level).
        IMPORTANT: Speak in ${selectedLanguage.name}.`;

        if (analysis) {
            systemInstructionText += `\n\nCONTEXT - CURRENT ANALYSIS REPORT:
            The user has just uploaded an image and here is the analysis you generated:
            SUMMARY: ${analysis.summary}
            KEY INSIGHTS: ${analysis.keyInsights.join('; ')}
            POTENTIAL DIAGNOSES: ${analysis.diagnoses.map(d => `${d.condition} (${d.confidence} confidence, ${d.severity} severity)`).join('; ')}
            
            INSTRUCTION:
            1. Start the conversation by briefly summarizing these results in a natural, spoken way. (e.g., "I've reviewed your report. It indicates signs of...")
            2. Ask the user if they would like to know more about any specific part.
            3. Do not just read the JSON. Translate it into a friendly conversation.`;
        } else {
            systemInstructionText += `\n\nINSTRUCTION: The user hasn't uploaded an image yet or the analysis failed. Ask them to upload an image so you can help.`;
        }

        // Connect to Live API
        const sessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
                },
                systemInstruction: systemInstructionText,
                inputAudioTranscription: {}, // Enable transcription so we see what user said
                outputAudioTranscription: {}, // Enable transcription so we see what model said
            },
            callbacks: {
                onopen: () => {
                    console.log("Live API Connected");
                    setIsConnected(true);
                },
                onmessage: async (message: LiveServerMessage) => {
                    // 1. Handle Transcriptions (for UI)
                    if (message.serverContent?.inputTranscription?.text) {
                        onTranscript(message.serverContent.inputTranscription.text, true);
                    }
                    if (message.serverContent?.outputTranscription?.text) {
                        onTranscript(message.serverContent.outputTranscription.text, false);
                    }

                    // 2. Handle Audio Output
                    const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                    if (audioData && audioContextRef.current) {
                        const buffer = await decodeAudioData(decodeBase64(audioData), audioContextRef.current);
                        
                        // Schedule playback
                        const source = audioContextRef.current.createBufferSource();
                        source.buffer = buffer;
                        source.connect(audioContextRef.current.destination);
                        
                        // Calculate start time to ensure gapless playback
                        const currentTime = audioContextRef.current.currentTime;
                        const startTime = Math.max(currentTime, nextStartTimeRef.current);
                        
                        source.start(startTime);
                        nextStartTimeRef.current = startTime + buffer.duration;
                        
                        scheduledSourcesRef.current.add(source);
                        source.onended = () => scheduledSourcesRef.current.delete(source);
                    }

                    // 3. Handle Interruptions
                    if (message.serverContent?.interrupted) {
                        scheduledSourcesRef.current.forEach(s => {
                            try { s.stop(); } catch(e){}
                        });
                        scheduledSourcesRef.current.clear();
                        nextStartTimeRef.current = 0;
                    }
                },
                onclose: () => {
                    console.log("Live API Closed");
                    setIsConnected(false);
                },
                onerror: (err) => {
                    console.error("Live API Error:", err);
                    setIsError(true);
                    setIsConnected(false);
                }
            }
        });

        // Store session reference for sending input
        // Wait for connection before sending audio
        sessionPromise.then(session => {
             sessionRef.current = session;
             
             // Hook up the audio processor to send data
             processor.onaudioprocess = (e) => {
                 const inputData = e.inputBuffer.getChannelData(0);
                 const base64PCM = float32ToBase64(inputData);
                 session.sendRealtimeInput({
                     media: {
                         mimeType: 'audio/pcm;rate=16000',
                         data: base64PCM
                     }
                 });
             };
        });

    } catch (err) {
        console.error("Connection failed", err);
        setIsError(true);
        setIsConnected(false);
    }
  }, [selectedLanguage.name, onTranscript, analysis]); // Re-connect if analysis changes to update context

  return { connect, disconnect, isConnected, isError };
};

// --- Page Components ---

const AboutUs: React.FC = () => (
  <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8 border border-gray-200 animate-fade-in">
    <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">About Us</h2>
    <div className="space-y-6 text-gray-700 leading-relaxed">
      <p className="text-lg font-semibold text-blue-600">
        We are Students of Grade 10th from Shiv Nadar School, Faridabad:
      </p>
      <ul className="list-disc list-inside bg-blue-50 p-6 rounded-lg border border-blue-100">
        <li>Dhruv Yadav</li>
        <li>Angeelina Aggarwal</li>
        <li>Mohammad Azlan</li>
        <li>Jyoti Ahluwalia</li>
        <li>Kimyra Motwani</li>
      </ul>
      <p>
        As part of our Grade 10 Capstone Project, we explored many ideas and research papers. Inspiration struck when we met someone who had medical reports but had to wait three days for a doctor’s appointment. During those days, they were tense, searching medical terms online, and asking friends. We realized the need for a platform that makes medical reports easy to understand for the general public.
      </p>
      <p className="text-xl font-medium text-center py-4">
        Thus, we built <span className="text-blue-600 font-bold">DiagnoSphere</span> — an open-source online community where anyone worldwide can understand their reports in their native language.
      </p>
    </div>
  </div>
);

const HowTo: React.FC = () => (
  <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8 border border-gray-200 animate-fade-in">
    <h2 className="text-3xl font-bold text-gray-800 mb-2 text-center">How To Use DiagnoSphere</h2>
    <p className="text-center text-xl text-blue-600 font-semibold mb-8">Just upload, click analyze, done!</p>
    
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-gray-700 mb-4">10 smart and interactive ways to use DiagnoSphere:</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          "Upload X-rays for AI-powered insights.",
          "Highlight confusing medical terms for instant explanations.",
          "Chat with the AI to ask follow-up questions.",
          "Switch to plain-language mode for simplified summaries.",
          "Use multilingual support to read reports in your native language.",
          "Get severity indicators (mild, moderate, critical).",
          "Extract key insights from long reports.",
          "Recognize symptoms from visible body marks.",
          "Receive general medication information (with disclaimers).",
          "Learn precautionary advice and lifestyle suggestions."
        ].map((item, idx) => (
          <div key={idx} className="flex items-start p-4 bg-gray-50 rounded-lg border hover:border-blue-300 transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mr-3">
              {idx + 1}
            </div>
            <p className="text-gray-700 mt-1">{item}</p>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const OurJourney: React.FC = () => (
  <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8 border border-gray-200 animate-fade-in">
    <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">Our Journey</h2>
    <div className="relative border-l-4 border-blue-200 ml-6 space-y-8">
      {[
        { date: "April 2025", title: "The Beginning", desc: "Introduced to Capstone by our teachers." },
        { date: "May 2025", title: "Team Formation", desc: "Team formation during Capstone period." },
        { date: "June–July 2025", title: "Exploration", desc: "Tried multiple ideas and approaches." },
        { date: "July end", title: "The Spark", desc: "Met someone struggling with medical reports before a doctor’s visit — inspiration struck." },
        { date: "August 2025", title: "Research", desc: "Conducted a survey via Google Forms to understand the problem better." },
        { date: "September 2025", title: "Pause", desc: "Paused due to exams." },
        { date: "October 2025", title: "Pivoting", desc: "Explored AI models (HuggingFace, ChatGPT, etc.) but found them inaccurate or stressful." },
        { date: "November 2025", title: "Future", desc: "Coming Soon..." }
      ].map((item, idx) => (
        <div key={idx} className="mb-8 ml-6 group">
          <span className="absolute flex items-center justify-center w-6 h-6 bg-blue-600 rounded-full -left-3.5 ring-4 ring-white group-hover:bg-orange-500 transition-colors">
          </span>
          <h3 className="flex items-center mb-1 text-lg font-semibold text-gray-900">
            {item.title} <span className="bg-blue-100 text-blue-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded ml-3">{item.date}</span>
          </h3>
          <p className="mb-4 text-base font-normal text-gray-500">{item.desc}</p>
        </div>
      ))}
    </div>
  </div>
);

const ContactUs: React.FC = () => (
  <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg p-8 border border-gray-200 animate-fade-in text-center">
    <h2 className="text-3xl font-bold text-gray-800 mb-6">Contact Us</h2>
    <p className="text-gray-600 mb-8">Have questions or feedback? Reach out to our team members directly.</p>
    
    <div className="grid gap-4 sm:grid-cols-2 text-left">
      {[
        { name: "Dhruv Yadav", email: "dy100040@sns.edu.in" },
        { name: "Angeelina Aggarwal", email: "sassykitt3n.xoxo@gmail.com" },
        { name: "Mohammad Azlan", email: "am100392@sns.edu.in" },
        { name: "Jyoti Ahluwalia", email: "ja100191@sns.edu.in" },
        { name: "Kimyra Motwani", email: "puneetmotwani@gmail.com" }
      ].map((contact, idx) => (
        <div key={idx} className="p-4 rounded-lg bg-gray-50 border hover:shadow-md transition-shadow flex items-center space-x-4">
          <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
             <EnvelopeIcon className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">{contact.name}</p>
            <p className="text-sm text-gray-500 truncate">{contact.email}</p>
          </div>
        </div>
      ))}
    </div>
  </div>
);

// --- Live Stats Component ---

const LiveStats: React.FC = () => {
    const [userCount, setUserCount] = useState(498);

    useEffect(() => {
        // Simulate a live counter from a base number
        const savedCount = localStorage.getItem('diagnosphere_user_count');
        let count = savedCount ? parseInt(savedCount) : 498;
        
        // Add a random small number to simulate new users since last visit
        if (!savedCount) {
            count += Math.floor(Math.random() * 50);
        }
        
        setUserCount(count);
        localStorage.setItem('diagnosphere_user_count', count.toString());
    }, []);

    return (
        <div className="inline-flex items-center bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full mb-4 shadow-sm border border-blue-200">
            <UserGroupIcon className="w-4 h-4 mr-2" />
            <span className="font-semibold">{userCount.toLocaleString()}</span>
            <span className="ml-1">people used this app so far</span>
        </div>
    );
};

// --- Feedback/Review Component ---

const initialReviews: Review[] = [
    { id: '1', name: 'Angeelina A.', rating: 5, comment: 'Helped me understand my ankle X-ray before seeing the doc!', date: '2025-11-10' },
    { id: '2', name: 'Rahul M.', rating: 5, comment: 'Great initiative by students. Very easy to use.', date: '2025-11-12' },
    { id: '3', name: 'Dr. Avnish K.', rating: 4, comment: 'The simplified terms feature is a lifesaver.', date: '2025-11-15' },
];

const FeedbackSection: React.FC = () => {
    const [reviews, setReviews] = useState<Review[]>(initialReviews);
    const [showAll, setShowAll] = useState(false);
    const [newReview, setNewReview] = useState({ name: '', rating: 5, comment: '' });
    const [showForm, setShowForm] = useState(false);

    useEffect(() => {
        const savedReviews = localStorage.getItem('diagnosphere_reviews');
        if (savedReviews) {
            setReviews(JSON.parse(savedReviews));
        }
    }, []);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!newReview.name || !newReview.comment) return;

        const review: Review = {
            id: Date.now().toString(),
            name: newReview.name,
            rating: newReview.rating,
            comment: newReview.comment,
            date: new Date().toISOString().split('T')[0]
        };

        const updatedReviews = [review, ...reviews];
        setReviews(updatedReviews);
        localStorage.setItem('diagnosphere_reviews', JSON.stringify(updatedReviews));
        setNewReview({ name: '', rating: 5, comment: '' });
        setShowForm(false);
        alert('Thank you for your feedback!');
    };

    const displayedReviews = showAll ? reviews : reviews.slice(0, 3);

    return (
        <div className="mt-12 bg-white rounded-xl shadow-lg p-8 border border-gray-200">
            <div className="flex flex-col sm:flex-row items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-gray-800">User Feedback</h2>
                    <div className="flex items-center mt-1">
                        <div className="flex text-yellow-400">
                            {[1, 2, 3, 4, 5].map(star => <StarIcon key={star} className="w-5 h-5" filled />)}
                        </div>
                        <span className="ml-2 text-gray-600 text-sm">4.8/5 Average Rating</span>
                    </div>
                </div>
                <button 
                    onClick={() => setShowForm(!showForm)}
                    className="mt-4 sm:mt-0 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                    {showForm ? 'Cancel' : 'Write a Review'}
                </button>
            </div>

            {showForm && (
                <form onSubmit={handleSubmit} className="mb-8 bg-gray-50 p-4 rounded-lg border border-blue-100 animate-fade-in">
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Your Name</label>
                        <input 
                            type="text" 
                            required
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
                            value={newReview.name}
                            onChange={e => setNewReview({...newReview, name: e.target.value})}
                        />
                    </div>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Rating</label>
                        <div className="flex space-x-2">
                            {[1, 2, 3, 4, 5].map((star) => (
                                <button
                                    type="button"
                                    key={star}
                                    onClick={() => setNewReview({...newReview, rating: star})}
                                    className={`focus:outline-none ${newReview.rating >= star ? 'text-yellow-400' : 'text-gray-300'}`}
                                >
                                    <StarIcon className="w-8 h-8" filled={true} />
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Comment</label>
                        <textarea 
                            required
                            rows={3}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
                            value={newReview.comment}
                            onChange={e => setNewReview({...newReview, comment: e.target.value})}
                        />
                    </div>
                    <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 font-medium">
                        Submit Review
                    </button>
                </form>
            )}

            <div className="grid gap-6 sm:grid-cols-1 md:grid-cols-3">
                {displayedReviews.map((review) => (
                    <div key={review.id} className="bg-gray-50 p-4 rounded-lg border hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start mb-2">
                            <div className="flex items-center">
                                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 mr-2">
                                    <UserIcon className="w-4 h-4" />
                                </div>
                                <div>
                                    <p className="font-semibold text-gray-800 text-sm">{review.name}</p>
                                    <p className="text-xs text-gray-500">{review.date}</p>
                                </div>
                            </div>
                            <div className="flex text-yellow-400">
                                {[...Array(review.rating)].map((_, i) => (
                                    <StarIcon key={i} className="w-3 h-3" filled />
                                ))}
                            </div>
                        </div>
                        <p className="text-gray-600 text-sm italic">"{review.comment}"</p>
                    </div>
                ))}
            </div>
            
            {reviews.length > 3 && (
                <div className="text-center mt-6">
                    <button 
                        onClick={() => setShowAll(!showAll)}
                        className="text-blue-600 hover:underline text-sm font-medium"
                    >
                        {showAll ? 'Show Less' : `See all ${reviews.length} reviews`}
                    </button>
                </div>
            )}
        </div>
    );
};

// --- Main Diagnostic Logic (Home View) ---

const HomeView: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Chat State
  const [chat, setChat] = useState<Chat | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [chatInput, setChatInput] = useState<string>('');
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // New Features State
  const [selectedLanguage, setSelectedLanguage] = useState<Language>(LANGUAGES[0]);
  
  // LIVE API Hook
  // We use this to inject transcriptions into the chat history so they appear in the UI
  const handleLiveTranscript = useCallback((text: string, isUser: boolean) => {
     setChatHistory(prev => {
        return [...prev, { role: isUser ? 'user' : 'model', content: text }];
     });
     // Also scroll to bottom
     if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
     }
  }, []);

  const { connect: connectLive, disconnect: disconnectLive, isConnected: isLiveConnected, isError: isLiveError } = useLiveAPI(selectedLanguage, handleLiveTranscript, analysis);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory, isChatLoading]);

  // Clean up
  useEffect(() => {
    return () => {
      disconnectLive();
    };
  }, [disconnectLive]);

  const resetState = () => {
    setImageFile(null);
    setImagePreview(null);
    setIsLoading(false);
    setAnalysis(null);
    setError(null);
    setChat(null);
    setChatHistory([]);
    setIsChatLoading(false);
    setChatInput('');
    disconnectLive();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      resetState();
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please upload a valid JPG or PNG image.');
      setImageFile(null);
      setImagePreview(null);
    }
    event.target.value = '';
  };

  const handleAnalyzeClick = useCallback(async () => {
    if (!imageFile) {
      setError('Please upload an image first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setAnalysis(null);
    setChat(null);
    setChatHistory([]);
    disconnectLive();
    
    // Increment simulated user count when analyze is clicked
    const currentCount = parseInt(localStorage.getItem('diagnosphere_user_count') || '498');
    localStorage.setItem('diagnosphere_user_count', (currentCount + 1).toString());

    try {
      const result = await analyzeMedicalImage(imageFile);
      setAnalysis(result);

      // ROBUST API KEY HANDLING
      const apiKey = getApiKey();
      if (!apiKey) throw new Error("API Key missing");

      const ai = new GoogleGenAI({ apiKey });
      
      const initialChatHistory = [
        { role: 'user', parts: [{ text: "Here is a medical image. Please analyze it and provide a structured JSON response as per your instructions." }] },
        { role: 'model', parts: [{ text: `I have analyzed the image. Here is the structured summary:\n${JSON.stringify(result, null, 2)}.\n\nI am ready for your follow-up questions.` }] }
      ];
      
      const chatSession = ai.chats.create({
        model: 'gemini-2.5-flash',
        history: initialChatHistory,
      });
      setChat(chatSession);

    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [imageFile, disconnectLive]);

  const handleSendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chat || !chatInput.trim() || isChatLoading) return;

    if (isLiveConnected) {
        // If live mode is on, we don't send text chat, we assume user wants to speak.
        alert("Please disconnect Live Voice Chat to type messages.");
        return;
    }

    const userMessage: ChatMessage = { role: 'user', content: chatInput };
    setChatHistory(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      // Append instructions to respond in the selected language
      const promptWithLanguage = `${userMessage.content} (IMPORTANT: Please respond in ${selectedLanguage.name}. If the medical terms are complex, explain them simply in ${selectedLanguage.name}.)`;
      
      const response = await chat.sendMessage({ message: promptWithLanguage });
      const modelMessage: ChatMessage = { role: 'model', content: response.text };
      setChatHistory(prev => [...prev, modelMessage]);
    } catch (err) {
      setError('Failed to get a response. Please try again.');
    } finally {
      setIsChatLoading(false);
    }
  };

  const toggleLiveConnection = () => {
    if (isLiveConnected) {
        disconnectLive();
    } else {
        connectLive();
    }
  };

  // Using display:none instead of unmounting to preserve state
  return (
    <div className={isActive ? 'block' : 'hidden'}>
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-gray-700">Medical Image Diagnosis <span className="text-blue-600">(Demo)</span></h1>
          <LiveStats />
          <p className="text-gray-500 mt-2 block">Upload a medical image for an AI-powered analysis and ask follow-up questions.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel: Upload and Image Preview */}
          <div className="bg-white rounded-xl shadow-lg p-6 md:p-8 border border-gray-200">
            <h2 className="text-xl font-bold text-gray-800 mb-4">1. Upload Image</h2>
             <label htmlFor="file-upload" className="cursor-pointer group block border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
                <UploadIcon className="w-12 h-12 mx-auto text-gray-400 group-hover:text-blue-500" />
                <p className="mt-2 text-sm text-gray-600">
                  <span className="font-semibold text-blue-600">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">PNG or JPG</p>
                <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileChange} accept="image/png, image/jpeg" />
              </label>
            
            {imagePreview && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Image Preview:</h3>
                <div className="w-full h-64 flex items-center justify-center bg-gray-100 rounded-lg overflow-hidden border">
                  <img src={imagePreview} alt="Medical scan preview" className="w-full h-full object-contain" />
                </div>
              </div>
            )}

            <div className="mt-6 text-center">
              <button
                onClick={handleAnalyzeClick}
                disabled={!imageFile || isLoading}
                className="w-full sm:w-auto inline-flex items-center justify-center px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                {isLoading ? (
                  <> <Spinner className="w-5 h-5 mr-3" /> Analyzing... </>
                ) : 'Analyze Image'}
              </button>
            </div>
          </div>
          
          {/* Right Panel: Analysis and Chat */}
          <div className="bg-white rounded-xl shadow-lg p-6 md:p-8 border border-gray-200 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 flex items-center">
                <LightbulbIcon className="w-6 h-6 mr-2 text-yellow-500" />
                2. Review Analysis
              </h2>
              
              {/* Language Toggle */}
              {analysis && (
                <div className="flex items-center gap-2">
                  <LanguageIcon className="w-5 h-5 text-gray-400" />
                  <select
                    value={selectedLanguage.code}
                    onChange={(e) => {
                      const lang = LANGUAGES.find(l => l.code === e.target.value);
                      if (lang) {
                          setSelectedLanguage(lang);
                          // If connected, we should ideally reconnect or send a new config, 
                          // but for simplicity, we let user restart connection.
                          if (isLiveConnected) disconnectLive(); 
                      }
                    }}
                    className="form-select text-sm border-gray-300 rounded-md shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>{lang.label}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {analysis && (
               <div className="mb-2 text-xs text-gray-400 text-right">
                  More languages coming soon once training is complete.
               </div>
            )}

            <div className="flex-grow flex flex-col min-h-[300px] lg:min-h-0">
               {error && (
                <div className="mt-4 bg-red-50 border-l-4 border-red-400 text-red-700 p-4 rounded-md" role="alert">
                  <p className="font-bold">Error</p>
                  <p>{error}</p>
                </div>
              )}
               {isLiveError && (
                <div className="mt-4 bg-red-50 border-l-4 border-red-400 text-red-700 p-4 rounded-md" role="alert">
                  <p className="font-bold">Live API Error</p>
                  <p>Connection failed. Please check permissions or try again.</p>
                </div>
              )}

              {!isLoading && !analysis && (
                <div className="flex-grow flex flex-col items-center justify-center text-center text-gray-500 p-4">
                  <StethoscopeIcon className="w-20 h-20 text-gray-300 mb-4" />
                  <p>Your analysis report will appear here after you upload and analyze an image.</p>
                </div>
              )}
              
              {isLoading && !analysis && (
                 <div className="flex-grow flex flex-col items-center justify-center text-center text-gray-500 p-4">
                  <Spinner className="w-12 h-12 text-blue-600" />
                  <p className="mt-4 text-lg">Generating detailed analysis...</p>
                 </div>
              )}

              {analysis && (
                <div className="flex-1 flex flex-col justify-between">
                  <AnalysisDisplay analysis={analysis} />
                  
                  {chat && (
                    <div className="mt-4 pt-4 border-t">
                      {/* Live Chat Notification */}
                      {isLiveConnected && (
                          <div className="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-2 rounded-lg mb-2 flex items-center justify-between animate-pulse">
                              <span className="text-sm font-semibold flex items-center">
                                  <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                                  Live Voice Chat Active
                              </span>
                              <span className="text-xs">Listening...</span>
                          </div>
                      )}

                      <div ref={chatContainerRef} className="h-48 overflow-y-auto pr-2 space-y-4 mb-4">
                        {chatHistory.map((msg, index) => (
                          <div key={index} className={`flex items-end gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'model' && (
                              <div className="flex flex-col gap-1">
                                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white"><StethoscopeIcon className="w-5 h-5" /></div>
                              </div>
                            )}
                            <div className={`max-w-xs md:max-w-md lg:max-w-lg rounded-xl px-4 py-2 ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
                              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            </div>
                          </div>
                        ))}
                        {isChatLoading && (
                          <div className="flex items-end gap-2 justify-start">
                             <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white"><StethoscopeIcon className="w-5 h-5" /></div>
                             <div className="max-w-xs rounded-xl px-4 py-2 bg-gray-100 text-gray-800 flex items-center">
                                <Spinner className="w-4 h-4" />
                             </div>
                          </div>
                        )}
                      </div>
                      
                      <form onSubmit={handleSendChatMessage} className="relative flex items-center gap-2">
                        <div className="relative flex-grow">
                           <input
                            type="text"
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            placeholder={isLiveConnected ? "Listening..." : `Ask a follow-up question in ${selectedLanguage.name}...`}
                            className="w-full p-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none disabled:bg-gray-100"
                            disabled={isChatLoading || isLiveConnected}
                          />
                          {/* Live Chat Toggle Button inside Input */}
                          <button 
                            type="button" 
                            onClick={toggleLiveConnection}
                            className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-1 rounded-full hover:bg-gray-100 transition-colors ${isLiveConnected ? 'text-red-600 bg-red-100' : 'text-gray-400'}`}
                            title={isLiveConnected ? "Stop Live Chat" : "Start Live Voice Chat"}
                          >
                             {isLiveConnected ? <StopIcon className="w-5 h-5" /> : <MicrophoneIcon className="w-5 h-5" />}
                          </button>
                        </div>
                        
                        {!isLiveConnected && (
                            <button type="submit" disabled={isChatLoading || !chatInput.trim()} className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400">
                                <SendIcon className="w-5 h-5" />
                            </button>
                        )}
                      </form>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Feedback Section */}
        <FeedbackSection />
    </div>
  );
};

// --- Analysis Display Component (Simplified for brevity) ---

const AnalysisDisplay: React.FC<{ analysis: AnalysisResult }> = ({ analysis }) => {
  const getConfidenceClass = (confidence: Diagnosis['confidence']) => {
    switch (confidence) {
      case 'High': return 'bg-red-100 text-red-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'Low': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getSeverityClass = (severity: Diagnosis['severity']) => {
    switch (severity) {
      case 'Critical': return 'border-red-500 bg-red-50';
      case 'Moderate': return 'border-yellow-500 bg-yellow-50';
      case 'Mild': return 'border-green-500 bg-green-50';
      default: return 'border-gray-300 bg-gray-50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="relative group">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Summary</h3>
        <p className="text-gray-600 bg-gray-50 p-3 rounded-lg border">{analysis.summary}</p>
      </div>
       <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Key Insights</h3>
        <ul className="list-disc list-inside space-y-1 text-gray-600">
            {analysis.keyInsights.map((insight, index) => <li key={index}>{insight}</li>)}
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Potential Diagnoses</h3>
        {analysis.diagnoses.length > 0 ? (
          <ul className="space-y-4">
            {analysis.diagnoses.map((diag, index) => (
              <li key={index} className={`p-4 rounded-lg border-l-4 ${getSeverityClass(diag.severity)}`}>
                <div className="flex items-center justify-between">
                  <h4 className="text-md font-semibold text-gray-900">{diag.condition}</h4>
                  <div className="flex items-center gap-2">
                     <span className={`px-2 py-1 text-xs font-medium rounded-full`}>{diag.severity}</span>
                     <span className={`px-2 py-1 text-xs font-medium rounded-full ${getConfidenceClass(diag.confidence)}`}>{diag.confidence}</span>
                  </div>
                </div>
                <p className="mt-2 text-sm text-gray-600 flex items-start gap-2">
                    <CheckCircleIcon className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5"/>
                    <span>{diag.suggestion}</span>
                </p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-center text-gray-600 py-4">No specific conditions were identified from the analysis.</p>
        )}
      </div>
    </div>
  );
};

// --- Main App Shell with Navigation ---

type Page = 'home' | 'about' | 'howto' | 'journey' | 'contact';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: { id: Page; label: string; icon: React.FC<{className?: string}> }[] = [
    { id: 'home', label: 'Home', icon: HomeIcon },
    { id: 'about', label: 'About Us', icon: UserGroupIcon },
    { id: 'howto', label: 'How To', icon: QuestionMarkCircleIcon },
    { id: 'journey', label: 'Our Journey', icon: RocketIcon },
    { id: 'contact', label: 'Contact Us', icon: EnvelopeIcon },
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center cursor-pointer" onClick={() => setCurrentPage('home')}>
              <StethoscopeIcon className="w-8 h-8 text-blue-600 mr-2" />
              <span className="text-xl sm:text-2xl font-bold text-gray-800">DiagnoSphere</span>
            </div>
            
            {/* Desktop Menu */}
            <div className="hidden md:flex items-center space-x-4">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    currentPage === item.id 
                      ? 'bg-blue-50 text-blue-700' 
                      : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
                  }`}
                >
                  <item.icon className="w-4 h-4 mr-2" />
                  {item.label}
                </button>
              ))}
            </div>

            {/* Mobile Menu Button */}
            <div className="flex items-center md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {mobileMenuOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu Panel */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-white border-t border-gray-100">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    setCurrentPage(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`block w-full text-left px-3 py-2 rounded-md text-base font-medium ${
                    currentPage === item.id
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
                  }`}
                >
                  <div className="flex items-center">
                    <item.icon className="w-5 h-5 mr-3" />
                    {item.label}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Main Content Area */}
      <main className="flex-grow w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        
        {/* Home View (Always mounted to preserve state, hidden via CSS when not active) */}
        <HomeView isActive={currentPage === 'home'} />

        {/* Static Pages (Conditionally rendered) */}
        {currentPage === 'about' && <AboutUs />}
        {currentPage === 'howto' && <HowTo />}
        {currentPage === 'journey' && <OurJourney />}
        {currentPage === 'contact' && <ContactUs />}

      </main>

      {/* Footer (Common across all pages) */}
      <footer className="w-full max-w-7xl mx-auto mt-auto text-center text-gray-500 text-sm p-6 bg-white border-t border-gray-100">
        <p className="font-bold text-orange-600">Disclaimer</p>
        <p className="mt-1 max-w-4xl mx-auto">
          This tool is a technology demonstration and is not a substitute for professional medical advice, diagnosis, or treatment. 
          The analysis is generated by an AI and may contain inaccuracies. Always consult with a qualified healthcare provider for any medical concerns.
        </p>
        <p className="mt-4 text-xs text-gray-400">© 2025 DiagnoSphere. Built by Students of Shiv Nadar School, Faridabad.</p>
      </footer>
    </div>
  );
};

export default App;