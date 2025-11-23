import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult } from '../types';

// Utility to convert a File object to a base64 string
const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]);
      } else {
        resolve('');
      }
    };
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};

export const analyzeMedicalImage = async (imageFile: File): Promise<AnalysisResult> => {
  const API_KEY = process.env.API_KEY;

  if (!API_KEY) {
    throw new Error("API Key is missing.");
  }
  const ai = new GoogleGenAI({ apiKey: API_KEY });

  const imagePart = await fileToGenerativePart(imageFile);

  const prompt = `
    You are a friendly, empathetic medical AI assistant for "DiagnoSphere", designed for the general public (patients, students, families).
    Your goal is to analyze medical images and explain the findings in **SIMPLE, EVERYDAY LANGUAGE** that a 10th-grade student can understand.
    
    **CRITICAL INSTRUCTION: NO MEDICAL JARGON WITHOUT TRANSLATION.**
    *   **FORBIDDEN TERMS (unless explained):** "reticular opacities", "peribronchial cuffing", "consolidation", "parenchyma", "mediastinum", "costophrenic angles", "silhouette".
    *   **REQUIRED TRANSLATION:**
        *   Instead of "reticular opacities," say "a pattern of lines or mesh-like marks (often signs of inflammation)."
        *   Instead of "peribronchial cuffing," say "thickening of the airway walls (often from a cold or infection)."
        *   Instead of "consolidation," say "fluid or swelling in the part of the lung."
        *   Instead of "blunted costophrenic angles," say "fluid collecting at the bottom of the lungs."
    
    **Primary Analysis Task:**
    Analyze the image and provide a structured JSON response:
    1.  **summary**: A clear, comforting paragraph summarizing what is wrong. **Address the user directly.** (e.g., "The scan shows signs of an infection in your lungs, likely pneumonia. The heart looks healthy.")
    2.  **keyInsights**: A bulleted list of observations in plain English. (e.g., "The lungs look a bit cloudy on both sides," rather than "Bilateral diffuse opacities").
    3.  **diagnoses**: An array of potential conditions.
        *   **condition**: The COMMON name (e.g., "Chest Infection" instead of "Lower Respiratory Tract Infection").
        *   **confidence**: 'High', 'Medium', 'Low', 'Uncertain'.
        *   **severity**: 'Critical', 'Moderate', 'Mild', 'Unknown'.
        *   **suggestion**: Simple, actionable advice (e.g., "See a doctor soon," "Rest and drink water").

    **Interaction Guidelines for Follow-up Chat:**
    *   Act as a "Translator" between doctor-speak and patient-speak.
    *   Always be kind and reassuring, but realistic.
    *   **Multi-lingual**: Respond in the language the user uses.

    **CRITICAL DISCLAIMER:** You are an AI. This analysis is NOT a diagnosis. Always consult a qualified healthcare provider.

    Now, please analyze the image and provide the structured JSON output.
    `;
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: { parts: [imagePart, { text: prompt }] },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: "A simple, easy-to-understand summary of the findings for a non-medical user."
            },
            keyInsights: {
              type: Type.ARRAY,
              items: {
                type: Type.STRING,
                description: "A simplified critical observation or takeaway."
              }
            },
            diagnoses: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  condition: {
                    type: Type.STRING,
                    description: "The name of the potential medical condition.",
                  },
                  confidence: {
                    type: Type.STRING,
                    description: "Confidence level (High, Medium, Low, Uncertain).",
                  },
                  severity: {
                    type: Type.STRING,
                    description: "Severity assessment (Critical, Moderate, Mild, Unknown)."
                  },
                  suggestion: {
                    type: Type.STRING,
                    description: "A simple, actionable suggestion for next steps.",
                  },
                },
                required: ["condition", "confidence", "severity", "suggestion"],
              },
            },
          },
          required: ["summary", "keyInsights", "diagnoses"],
        },
      },
    });

    const jsonText = response.text.trim();
    const parsedResult = JSON.parse(jsonText);
    return parsedResult as AnalysisResult;

  } catch (error) {
    console.error("Error analyzing image with Gemini API:", error);
    throw new Error("Failed to get a valid analysis from the AI. The model may be unable to process this image or there was a network issue.");
  }
};