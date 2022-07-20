import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress netAddress;


void setup () {
  oscP5 = new OscP5(this, 5005);
}

void draw() {

}


/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage msg) {
  /* print the address pattern and the typetag of the received OscMessage */
  println("### received an osc message.");
  println(" addrpattern: "+ msg.addrPattern());
  
  if(msg.checkAddrPattern("/dennis")){
    println(msg.get(0));
    println("Happened");
  }
  
  
  
}
