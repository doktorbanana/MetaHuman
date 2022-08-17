class MoverSystem{
  
  
ArrayList<Mover> movers = new ArrayList<Mover>();
PVector position;
Mover center;
float amp, radius;
float threshold = 0.1;
String text, name;
boolean drawLine;
PFont font;

  MoverSystem(String nam, PVector pos, int size, float grav, float centerGrav, float rad, boolean lineStyle){
     
    position = pos;
    center = new Mover(new PVector(0,0), centerGrav, new PVector(0,0));
    radius = rad;
    amp=0.01;
    text = "";
    drawLine = lineStyle;
    name = nam;
    font = createFont("Liberation Mono", 14, true);
    
    for(int i = 0; i<size; i++){
      PVector posMover, velMover;
      
      posMover = new PVector(random(-width/2,width/2), random(-height/2,height/2));
      posMover.normalize();
      posMover.mult(random(50,150));
      
      velMover = new PVector(random(0,0.7), random(0,0.7));
      
      Mover mover = new Mover(posMover, grav, velMover);
      movers.add(mover);
    }
  }
  
  void display(){
    translate(position.x, position.y);
    //center.display();

    for(int i=0; i< movers.size(); i++){
      Mover mover = movers.get(i);
      mover.amp = amp;
      mover.update();
      center.attract(mover);
      
      
      // CHECK AMPLITUDE -> Move towards center if quiet
      if(amp < threshold){
        PVector dist;
        dist = PVector.sub(center.position, mover.position);
        if(dist.mag() > radius){
          dist.normalize();
          dist.mult(0.3);
          mover.scaledVel.add(dist);
        }
      }
      
      // DRAW THE SYSTEM
      stroke(200);
      strokeWeight(1);
      fill(0);
      
      if(drawLine){
        line(mover.position.x, mover.position.y, center.position.x, center.position.y);
      }else{
        //stroke(255);
        //strokeWeight(6);
        //fill(255);
        center.display();
        //stroke(200);
        //strokeWeight(1);
        //fill(0);
        mover.display();
        if(i<movers.size()-1){
          line(mover.position.x, mover.position.y, movers.get(i+1).position.x, movers.get(i+1).position.y);
        }else{
          line(mover.position.x, mover.position.y, center.position.x, center.position.y);
        }
      }
      
      for(Mover other : movers){
        if(mover != other){
          mover.attract(other);
          if(drawLine){
            line(mover.position.x, mover.position.y, other.position.x, other.position.y);
          }
        }
        
      }
    }
    
    fill(255);
    textAlign(CENTER);
    textFont(font);
    text(text, 0, radius * 3);
    text(name + ":", 0, (radius*3)-15);
    //imageMode(CENTER);
    //image(eye,0,0,radius,radius);
    
    translate(-position.x, -position.y); 
  }
}
