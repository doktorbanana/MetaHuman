import oscP5.*;
import netP5.*;
import processing.sound.*;

OscP5 oscP5;
NetAddress netAddress;

AudioIn in;
Amplitude amp;
float gain = 10;

MoverSystem valerio;
MoverSystem dennis;

String lastAddress;
PFont font;
PFont smallerFont;

String iteration;

String band;

void setup () {
  //size(1920,1080);
  fullScreen();
  oscP5 = new OscP5(this, 5005);
  valerio = new MoverSystem("Valerio", new PVector(width * 0.25, height * 0.55), 10, 1, 40, 50, false);
  dennis = new MoverSystem("Dennis", new PVector(width * 0.75, height * 0.55), 10, 1, 40, 50, true);
  
  font = createFont("Liberation Mono", 25, true);
  
  amp = new Amplitude(this);
  in = new AudioIn(this, 0);
  in.start();
  amp.input(in);
  
  lastAddress = "valerio";
}

void draw() {
  background(0);
  valerio.display();
  dennis.display();
  
  fill(255);
  textAlign(CENTER);
  textFont(font);
  text("Metahuman", width/2, height/8);
  textFont(font, 14);
  text("Machine Music based on a " + band + " song.\n" + "Iteration " + iteration, width/2, height/8 + 30);
  
  switch(lastAddress){
    
    case "Valerio":
      valerio.amp = map((amp.analyze()*gain), 0,gain, 0.02, 0.5);
      dennis.amp = 0.001;
      break;
    case "Dennis":
      dennis.amp = map((amp.analyze()*gain), 0,gain, 0.02, 0.5);
      valerio.amp = 0.001;
      break;
   } 
 }

/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage msg) {
  /* print the address pattern and the typetag of the received OscMessage */
  println(" addrpattern: "+ msg.addrPattern());
  println(msg.get(0));

  
  switch(msg.addrPattern().toString()){
  
    case "/Dennis":
      lastAddress = "Dennis";
      dennis.text = msg.get(0).toString();
      break;
    case "/Valerio":
      lastAddress = "Valerio";
      valerio.text = msg.get(0).toString();
      break;
    
    case "/DennisWaits":
      dennis.text = msg.get(0).toString();
      break;
      
    case "/ValerioWaits":
      valerio.text = msg.get(0).toString();
      break;
      
    case "/DennisSleeps":
      dennis.text = msg.get(0).toString();
      valerio.text = "";
      
      break;
      
    case "/ValerioSleeps":
      valerio.text = msg.get(0).toString();
      dennis.text = "";
      break;
    
    case "/ValerioDone":
      valerio.text = msg.get(0).toString();
      valerio.amp = 0.001;
      lastAddress="";
      break;
    case "/DennisDone":
      dennis.text = msg.get(0).toString();
      dennis.amp = 0.001;
      lastAddress="";
      break;
    case "/iteration":
      iteration = msg.get(0).toString();
      break;
    case "/DennisBand":
      band = "Beatles";
      break;
    case "ValerioBand":
      band = "Led Zeppelin";
      break;
  }
  
  
  
  
  
}
