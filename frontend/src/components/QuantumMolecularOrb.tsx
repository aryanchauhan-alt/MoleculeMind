import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface QuantumMolecularOrbProps {
  className?: string;
  particleCount?: number;
  interactive?: boolean;
}

export const QuantumMolecularOrb: React.FC<QuantumMolecularOrbProps> = ({
  className = "w-full h-full",
  particleCount = 1400,
  interactive = true
}) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 600;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 20;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Main Group for unified rotation & parallax
    const orbGroup = new THREE.Group();
    scene.add(orbGroup);

    // 1. Refined Minimal Crystalline Icosahedron Structure
    const coreGeometry = new THREE.IcosahedronGeometry(4.6, 2);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.08
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    orbGroup.add(coreMesh);

    // 2. Soft Inner Molecular Core Sphere
    const nucleusGeometry = new THREE.SphereGeometry(1.8, 32, 32);
    const nucleusMaterial = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.12,
      wireframe: true
    });
    const nucleusMesh = new THREE.Mesh(nucleusGeometry, nucleusMaterial);
    orbGroup.add(nucleusMesh);

    // 3. Ethereal Valence Particle Cloud (Silver, Ice Blue, and Soft Emerald)
    const particlesGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const silverColor = new THREE.Color(0xf1f5f9);
    const iceColor = new THREE.Color(0x38bdf8);
    const emeraldColor = new THREE.Color(0x34d399);

    for (let i = 0; i < particleCount; i++) {
      // Uniform spherical Fibonacci distribution
      const phi = Math.acos(1 - 2 * (i + 0.5) / particleCount);
      const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
      const radius = 6.2 + (Math.random() - 0.5) * 2.2;

      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Subtle editorial palette distribution
      const mixedColor = new THREE.Color();
      const p = Math.random();
      if (p < 0.55) {
        mixedColor.copy(silverColor);
      } else if (p < 0.85) {
        mixedColor.copy(iceColor);
      } else {
        mixedColor.copy(emeraldColor);
      }

      colors[i * 3] = mixedColor.r;
      colors[i * 3 + 1] = mixedColor.g;
      colors[i * 3 + 2] = mixedColor.b;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.14,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending
    });

    const particleSystem = new THREE.Points(particlesGeometry, particlesMaterial);
    orbGroup.add(particleSystem);

    // 4. Hairline Orbital Geodesic Rings
    const createRing = (radius: number, colorHex: number, rotX: number, rotY: number, opacity: number) => {
      const ringGeometry = new THREE.TorusGeometry(radius, 0.02, 16, 120);
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: opacity,
        wireframe: true
      });
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.rotation.x = rotX;
      ring.rotation.y = rotY;
      return ring;
    };

    const ring1 = createRing(6.8, 0xffffff, Math.PI / 4, 0, 0.2);
    const ring2 = createRing(7.3, 0x38bdf8, -Math.PI / 3, Math.PI / 6, 0.25);
    const ring3 = createRing(7.8, 0x34d399, Math.PI / 6, Math.PI / 3, 0.18);

    orbGroup.add(ring1);
    orbGroup.add(ring2);
    orbGroup.add(ring3);

    // 5. Small Satellite Atoms
    const nodeGeometry = new THREE.SphereGeometry(0.18, 16, 16);
    const nodeMaterial1 = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const nodeMaterial2 = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const nodeMaterial3 = new THREE.MeshBasicMaterial({ color: 0x34d399 });

    const node1 = new THREE.Mesh(nodeGeometry, nodeMaterial1);
    const node2 = new THREE.Mesh(nodeGeometry, nodeMaterial2);
    const node3 = new THREE.Mesh(nodeGeometry, nodeMaterial3);

    orbGroup.add(node1);
    orbGroup.add(node2);
    orbGroup.add(node3);

    // Mouse Parallax & Smooth Damping
    let targetRotationX = 0;
    let targetRotationY = 0;
    let mouseX = 0;
    let mouseY = 0;

    const handleMouseMove = (event: MouseEvent) => {
      if (!interactive) return;
      const rect = container.getBoundingClientRect();
      mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseY = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      targetRotationY = mouseX * 0.6;
      targetRotationX = -mouseY * 0.6;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // Responsive Resize
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    // Animation Loop with Fluid Breathing Physics
    let animationFrameId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Fluid continuous rotation
      orbGroup.rotation.y += 0.0035;
      orbGroup.rotation.x += 0.0015;

      // Smooth mouse parallax damping
      orbGroup.rotation.y += (targetRotationY - orbGroup.rotation.y) * 0.035;
      orbGroup.rotation.x += (targetRotationX - orbGroup.rotation.x) * 0.035;

      // Ethereal subtle breathing scale
      const breathScale = 1.0 + Math.sin(elapsedTime * 0.8) * 0.025;
      particleSystem.scale.set(breathScale, breathScale, breathScale);

      // Ring rotations
      ring1.rotation.z += 0.002;
      ring2.rotation.z -= 0.003;
      ring3.rotation.z += 0.0015;

      // Satellite nodes orbiting
      const t1 = elapsedTime * 0.7;
      node1.position.set(Math.cos(t1) * 6.8, Math.sin(t1) * 6.8 * 0.6, Math.sin(t1) * 6.8 * 0.8);

      const t2 = -elapsedTime * 0.5 + 2.0;
      node2.position.set(Math.sin(t2) * 7.3 * 0.7, Math.cos(t2) * 7.3, Math.cos(t2) * 7.3 * 0.5);

      const t3 = elapsedTime * 0.4 + 4.0;
      node3.position.set(Math.cos(t3) * 7.8 * 0.8, Math.sin(t3) * 7.8 * 0.5, Math.cos(t3) * 7.8);

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (container && renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      nucleusGeometry.dispose();
      nucleusMaterial.dispose();
      particlesGeometry.dispose();
      particlesMaterial.dispose();
    };
  }, [particleCount, interactive]);

  return (
    <div 
      ref={mountRef} 
      className={`relative cursor-grab active:cursor-grabbing ${className}`}
      style={{ touchAction: 'none' }}
    />
  );
};
