import Particles, { initParticlesEngine } from "@tsparticles/react";
import { useEffect, useMemo, useState } from "react";
import { loadSlim } from "@tsparticles/slim";

const ParticlesComponent = ({ id, size = 0 }) => {
  const [init, setInit] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => {
      setInit(true);
    });
  }, []);

  const particlesLoaded = (container) => {
    console.log(container);
  };

  const options = useMemo(
      () => ({
        background: {
          color: {
            value: "#0a0a0a",
          },
        },
        fpsLimit: 360,
        interactivity: {
          events: {
            onClick: {
              enable: true,
              mode: "repulse",
            },
            onHover: {
              enable: true,
              mode: "grab",
            },
          },
          modes: {
            repulse: {
              distance: 100, // smaller repulse range
              duration: 0.4,
              easing: "ease-out-quad",
              factor: 100,
              speed: 1,
              maxSpeed: 50,
              clamp: true, // prevent particles from being pushed out of bounds
            },
            grab: {
              distance: 150,
              links: {
                opacity: 0.5,
              },
            },
          },
        },
        particles: {
          color: {
            value: "#FFFFFF",
          },
          links: {
            color: "rgba(255,255,255,0)",
            distance: 150,
            enable: true,
            opacity: 0.8,
            width: 1.6,
          },
          move: {
            direction: "none",
            enable: true,
            outModes: {
              default: "bounce",
            },
            random: true,
            speed: 2.5,
            straight: false,
          },
          number: {
            density: {
              enable: true,
            },
            value: 300,
          },
          opacity: {
            value: 1.0,
          },
          shape: {
            type: "circle",
          },
          size: {
            value: size,
          },
        },
        detectRetina: true,
      }),
      [size]
  );

  if (!init) return null;

  return <Particles id={id} init={particlesLoaded} options={options} />;
};

export default ParticlesComponent;
