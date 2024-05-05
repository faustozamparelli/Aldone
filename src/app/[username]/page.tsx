import VoiceAI from "@/components/voice-ai";

export default function Page({ params }: { params: { username: string } }) {
  return (
    <div
      className="flex min-h-screen flex-col items-center justify-between"
      style={{
        background: "linear-gradient(to top, #B6CECE, #0099FF)",
      }}
    >
      <p className="text-white font-light text-xl animate-fadeInUp text-right w-full p-3">
        <span className="text-lime-200">{params.username}</span>
      </p>
      <div className="text-white">
        <VoiceAI />
      </div>
      <footer></footer>
    </div>
  );
}
