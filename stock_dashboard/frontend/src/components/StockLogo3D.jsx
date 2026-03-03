import { Suspense, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Environment, Clone } from '@react-three/drei';

/* ── The GLB model ───────────────────────────────────────── */
function Model({ scale = 1 }) {
    const { scene } = useGLTF('/stock-arrow.glb');
    return (
        <Clone
            object={scene}
            scale={scale}
            position={[0, 0, 0]}
        />
    );
}

/* ── Fallback spinner while model loads ─────────────────── */
function Loader() {
    return (
        <mesh>
            <sphereGeometry args={[0.3, 16, 16]} />
            <meshStandardMaterial color="#388bfd" wireframe />
        </mesh>
    );
}

/* ── Exported component ─────────────────────────────────── */
/**
 * Props:
 *   size       – pixel size of the canvas (square). Default 48
 *   modelScale – scale passed to the GLB primitive. Default 1.2
 *   autoRotate – whether the model spins on its own. Default true
 *   drag       – whether the user can orbit/drag. Default true
 *   className  – optional extra className on the wrapper div
 *   style      – optional inline styles on the wrapper div
 */
export default function StockLogo3D({
    size = 48,
    modelScale = 1.2,
    autoRotate = true,
    drag = true,
    className = '',
    style = {},
}) {
    return (
        <div
            className={className}
            style={{
                width: size,
                height: size,
                flexShrink: 0,
                cursor: drag ? 'grab' : 'default',
                borderRadius: '10px',
                overflow: 'hidden',
                ...style,
            }}
        >
            <Canvas
                camera={{ position: [0, 0, 2.5], fov: 45 }}
                gl={{ antialias: true, alpha: true }}
                style={{ background: 'transparent' }}
            >
                {/* Lighting */}
                <ambientLight intensity={1.0} />
                <directionalLight position={[5, 5, 5]} intensity={1.5} color="#ffffff" />
                <directionalLight position={[-5, -5, 2]} intensity={0.5} color="#bc8cff" />
                <pointLight position={[0, 1, 2]} intensity={1.2} color="#388bfd" />
                <pointLight position={[0, -2, 2]} intensity={0.6} color="#3fb950" />

                {/* Model */}
                <Suspense fallback={<Loader />}>
                    <Model scale={modelScale} />
                    <Environment preset="night" />
                </Suspense>

                {/* Controls */}
                {drag && (
                    <OrbitControls
                        enableZoom={false}
                        enablePan={false}
                        autoRotate={autoRotate}
                        autoRotateSpeed={3}
                        minPolarAngle={Math.PI / 6}
                        maxPolarAngle={Math.PI - Math.PI / 6}
                    />
                )}
                {!drag && (
                    <OrbitControls
                        enableZoom={false}
                        enablePan={false}
                        enableRotate={false}
                        autoRotate={autoRotate}
                        autoRotateSpeed={3}
                    />
                )}
            </Canvas>
        </div>
    );
}

// Pre-load the model so it's cached
useGLTF.preload('/stock-arrow.glb');
