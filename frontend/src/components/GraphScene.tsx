import { Canvas } from '@react-three/fiber';
import { OrbitControls, Float, Line, Text } from '@react-three/drei';
import { useMemo } from 'react';

type Node = {id:string;label:string;type:string;detail?:string}; type Edge={source:string;target:string;label?:string};
function SphereNode({node,position,active,onSelect}:{node:Node;position:[number,number,number];active:boolean;onSelect:()=>void}){
 const color=node.type==='policy'?'#c8f169':node.type==='risk'?'#fb7185':'#79e6bd';
 return <group position={position} onClick={(e)=>{e.stopPropagation();onSelect();}}>
  <Float speed={1.2} rotationIntensity={0.12} floatIntensity={0.16}><mesh><icosahedronGeometry args={[active?0.22:0.15,1]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={active?0.8:0.18} roughness={0.32}/></mesh></Float>
  <Text position={[0,-0.34,0]} fontSize={0.12} color="#e7f0eb" anchorX="center" maxWidth={1.7}>{node.label.slice(0,23)}</Text>
 </group>
}
export default function GraphScene({nodes,edges,selected,onSelect}:{nodes:Node[];edges:Edge[];selected:string|null;onSelect:(n:Node)=>void}){
 const positions=useMemo(()=>nodes.map((n,i)=>{const a=i*2.399; const r=nodes.length<3?0.7:0.75+0.28*(i%3); return {node:n,p:[Math.cos(a)*r,Math.sin(a)*r,Math.sin(i*1.7)*0.45] as [number,number,number]};}),[nodes]);
 const posById=new Map(positions.map(x=>[x.node.id,x.p]));
 return <div className="graph-canvas"><Canvas camera={{position:[0,0,4.5],fov:48}}><ambientLight intensity={1.2}/><pointLight position={[2,3,4]} intensity={18} color="#a7f3d0"/>
 {edges.map((e,i)=>{const a=posById.get(e.source),b=posById.get(e.target);return a&&b?<Line key={i} points={[a,b]} color="#537a6a" transparent opacity={0.6} lineWidth={1}/> : null;})}
 {positions.map(({node,p})=><SphereNode key={node.id} node={node} position={p} active={selected===node.id} onSelect={()=>onSelect(node)}/>)}
 <OrbitControls enablePan={false} minDistance={3} maxDistance={7}/></Canvas><div className="graph-hint">DRAG TO ORBIT · SCROLL TO ZOOM</div></div>
}