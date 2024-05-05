export default function Page({ params }: { params: { slug: string } }) {
  return (
    <div
      className="flex min-h-screen flex-col items-center justify-between"
      style={{
        background: "linear-gradient(to top, #B6CECE, #0099FF)",
      }}
    >
      <p className="m-auto text-white font-light text-3xl">
        {`Welcome back, `}
        <span className="text-lime-200">{params.slug}</span>
      </p>
    </div>
  );
}
