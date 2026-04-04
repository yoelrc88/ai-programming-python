export const callGeminiVision = async (base64Image, apiKey) => {
  try {
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg);base64,/, "");
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            role: "user",
            parts: [
              { text: `
                Analyze this book page image.
                1. Transcribe the text exactly.
                2. Create a simplified summary (50% length).
                3. Translate the full text to Spanish.
                4. Analyze the VISUAL STYLE. If text only, suggest a genre style.
                5. Extract keywords.
                Return ONLY raw JSON: { "fullText": "...", "simplified": "...", "spanish": "...", "visualStyle": "...", "searchQueries": ["..."] }
              `},
              { inlineData: { mimeType: "image/jpeg", data: cleanBase64 } }
            ]
          }],
          generationConfig: { responseMimeType: "application/json" }
        })
      }
    );
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    const data = await response.json();
    return JSON.parse(data.candidates[0].content.parts[0].text);
  } catch (error) {
    console.error("Gemini Vision Error:", error);
    throw error;
  }
};

export const generateImage = async (prompt, style, apiKey) => {
  try {
    const finalPrompt = `An artistic illustration of this scene: "${prompt}". The art style should be: ${style}. NO TEXT.`;
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instances: [{ prompt: finalPrompt }],
          parameters: { sampleCount: 1 }
        })
      }
    );
    if (!response.ok) throw new Error(`Imagen Error: ${response.status}`);
    const data = await response.json();
    return `data:image/png;base64,${data.predictions[0].bytesBase64Encoded}`;
  } catch (error) {
    console.error("Imagen Error:", error);
    return null;
  }
};
