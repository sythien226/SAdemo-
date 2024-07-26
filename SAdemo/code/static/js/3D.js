
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/loaders/GLTFLoader.js";

///////////////////////////////////////////////////////////////////////////////////////
// Init
const w = window.innerWidth;
const h = window.innerHeight;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, w / h, 10, 10000);
camera.position.set(0, 40, 400);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(w, h);
document.getElementById('3D').appendChild(renderer.domElement);

///////////////////////////////////////////////////////////////////////////////////////
// Control
const controls = new OrbitControls(camera, renderer.domElement);
controls.minDistance = 200;
controls.maxDistance = 1000;
controls.enableZoom = true;
controls.enableRotate = true;
controls.enablePan = false;
controls.update();

////////////////////////////////////////////////////////////////////////////////////////
const bgloader = new THREE.TextureLoader();
scene.background = bgloader.load("../static/texture/bg.jpg");
///////////////////////////////////////////////////////////////////////////////////////
// Light
const SunLight = new THREE.PointLight(0xffffff, 3, 2000);
SunLight.position.set(50, 50, 800);
scene.add(SunLight);
////////////////////////////////////////////////////////////////////////////////////
//sun object
const sunGeometry = new THREE.SphereGeometry(5, 32, 32);
// const sunMaterial = new THREE.MeshBasicMaterial({ color: 0xFFFF00 });
const textureSun = new THREE.TextureLoader().load("../static/texture/sun.jpg");
const sunMaterial = new THREE.MeshBasicMaterial({ map: textureSun });
const sun = new THREE.Mesh(sunGeometry, sunMaterial);
sun.position.x = SunLight.position.x;
sun.position.y = SunLight.position.y;
sun.position.z = SunLight.position.z;
scene.add(sun);
sun.scale.set(10, 10, 10);
////////////////////////////////////////////////////////////////////////////////////////
//Create texture
const stars_loader = new THREE.TextureLoader();
const starTexture = stars_loader.load('../static/texture/stars.jpg'); 
// Create stars
const starGeometry = new THREE.BufferGeometry();
const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, 
  size: 1.5,
  sizeAttenuation: true,
  map: starTexture,
  alphaTest: 0.5,
  transparent: true,
  depthWrite: false});

const starsVertices = [];
starsVertices.push(1.0, 1.0, 1.0);
for (let i = 0; i < 10000; i++) {
    const x = THREE.MathUtils.randFloatSpread(2000); // random
    const y = THREE.MathUtils.randFloatSpread(2000);
    const z = THREE.MathUtils.randFloatSpread(2000);
    starsVertices.push(x, y, z);
}

starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
const stars = new THREE.Points(starGeometry, starMaterial);
scene.add(stars);
////////////////////////////////////////////////////////////////////////////////////////////
// Face
// const frontMaterial = new THREE.MeshStandardMaterial({
//   map: new THREE.TextureLoader().load("../static/assets/diffuseMap_0.png"),
//   envMap: new THREE.TextureLoader().load("../static/assets/envMap_0.png"),
//   roughnessMap: new THREE.TextureLoader().load("../static/assets/roughnessMap_0.png"),
//   metalnessMap: new THREE.TextureLoader().load("../static/assets/specularMap_0.png"),
//   side: THREE.FrontSide
// });

// const backMaterial = new THREE.MeshPhongMaterial({
//   color: 0x0F0F0F,  // Màu đen
//   side: THREE.BackSide
// });

// let position_face = new THREE.Vector3();
// const loader = new OBJLoader();
// loader.load("../static/assets/mesh0.obj", (obj) => {
//   const geometry = obj.children[0].geometry;
//   const frontMesh = new THREE.Mesh(geometry, frontMaterial);
//   const backMesh = new THREE.Mesh(geometry, backMaterial);

//   // Để đảm bảo rằng backMesh hiển thị đúng ngay cả khi có sự chồng chất,
//   // ta sẽ đặt renderOrder để xử lý trình tự vẽ.
//   backMesh.renderOrder = 0;  // Vẽ trước
//   frontMesh.renderOrder = 1;  // Vẽ sau

//   scene.add(backMesh);
//   scene.add(frontMesh);

//   const std = ~~obj.children[0].geometry.attributes.position.array[2];
//   position_face.addScalar(0, 0, std);
//   frontMesh.position.set(0, 0, std);
//   frontMesh.rotation.z = -1 * Math.PI;
//   frontMesh.rotation.y = Math.PI;
//   backMesh.position.set(0, 0, std);
//   backMesh.rotation.z = -1 * Math.PI;
//   backMesh.rotation.y = Math.PI;

// });


// head
let position_face = new THREE.Vector3();
const loaderface = new GLTFLoader();

loaderface.load(
  '../static/assets/gabrielle_hersh.glb', 
  function (gltf) {
    scene.add(gltf.scene);
    gltf.scene.position.set(0, -135, 0);
    position_face.addScalar(0, -135, 0);
    gltf.scene.scale.set(2, 2, 2);
    console.log('Load successfully');
  },
  function (xhr) {
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
  },
  function (error) {
    console.log('An error happened');
  }
);
///////////////////////////////////////////////////////////////////
// Zoom
function handleWindowResize () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', handleWindowResize, false);
///////////////////////////////////////////////////////////////////////////////////////
// Animate
let angle = 0;
let rotation = 0;
function animate() {
  requestAnimationFrame(animate);

  // Update the stars
  stars.rotation.x += 0.001;
  stars.rotation.y += 0.001;
  // console.log(stars);

  // Update the sun light position
  angle += 0.005;
  rotation += 0.0007;
  SunLight.position.x = (position_face.x + 1500 * Math.cos(angle) * Math.sin(rotation));
  SunLight.position.z = position_face.z + 1000 * Math.sin(angle);
  SunLight.position.y = -(position_face.y + 1500 * Math.cos(angle) * Math.cos(rotation));

  // Update the sun position
  sun.rotation.x += 0.01;
  sun.rotation.y += 0.01;
  sun.position.x = SunLight.position.x;
  sun.position.z = SunLight.position.z;
  sun.position.y = SunLight.position.y;

  renderer.render(scene, camera);
}
/////////////////////////////////////////////////////////////////////////////////////////
animate();