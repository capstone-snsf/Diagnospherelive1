export interface Diagnosis {
  condition: string;
  confidence: 'High' | 'Medium' | 'Low' | 'Uncertain';
  suggestion: string;
  severity: 'Critical' | 'Moderate' | 'Mild' | 'Unknown';
}

export interface AnalysisResult {
  summary: string;
  keyInsights: string[];
  diagnoses: Diagnosis[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}
