import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Float, Line } from '@react-three/drei';

const points: { p: [number, number, number]; size: number; color: string }[] = [
  { p: [0.2, 0.5, 0], size: .09, color: '#c8f169' },
  { p: [1, .15, -.1], size: .07, color: '#79e6bd' },
  { p: [1.7, .6, -.2], size: .085, color: '#c8f169' },
  { p: [2.3, .1, .05], size: .06, color: '#79e6bd' },
  { p: [.65, -.45, .1], size: .07, color: '#79e6bd' },
  { p: [1.4, -.5, 0], size: .11, color: '#c8f169' },
  { p: [2.1, -.45, -.1], size: .075, color: '#79e6bd' },
  { p: [2.8, 0, .1], size: .09, color: '#c8f169' },
  { p: [1.95, 1.1, .1], size: .05, color: '#79e6bd' },
  { p: [.9, 1.05, -.2], size: .05, color: '#c8f169' },
  { p: [2.8, -.85, 0], size: .05, color: '#79e6bd' },
];
const links: [number, number][] = [[0,1],[0,4],[0,9],[1,2],[1,4],[1,5],[2,3],[2,5],[2,8],[3,7],[3,6],[4,5],[5,6],[5,7],[6,7],[6,10],[8,9]];

function NetworkScene() {
  const lines = useMemo(() => links.map(([a,b]) => [points[a].p, points[b].p] as [[number,number,number],[number,number,number]]), []);
  return <>
    <ambientLight intensity={1.15} />
    <pointLight position={[2,2,4]} intensity={7} color="#c8f169" />
    {lines.map((line,i) => <Line key={i} points={line} color="#709581" transparent opacity={.28} lineWidth={.8} />)}
    {points.map((point,i) => <Float key={i} speed={.45+(i%4)*.12} rotationIntensity={.08} floatIntensity={.08}>
      <mesh position={point.p}><icosahedronGeometry args={[point.size,1]} /><meshStandardMaterial color={point.color} emissive={point.color} emissiveIntensity={.45} roughness={.35} /></mesh>
    </Float>)}
  </>;
}

export default function KnowledgeGraphBackground() {
  return <div className="cs-knowledge-graph" aria-hidden="true">
    <Canvas camera={{ position: [1.5,.15,5.8], fov: 43 }} dpr={[1,1.35]} gl={{ alpha: true, antialias: true }}><NetworkScene /></Canvas>
    <div className="cs-graph-fade" />
    <div className="cs-graph-label cs-graph-label-one">POLICY LINK <span>CONNECTED</span></div>
    <div className="cs-graph-label cs-graph-label-two">CLAUSE NODE <span>ANALYSIS</span></div>
    <div className="cs-graph-label cs-graph-label-three">RISK SIGNAL <span>REVIEW</span></div>
  </div>;
}
