import processing.sound.*;
class Mover{
  
  PVector position, velocity, acceleration;
  PVector scaledVel;
  float mass, grav, distanceFactor;
  float amp;
  float maxVelo = 3;
  
  
  Mover(PVector pos, float g, PVector vel){
    position = pos;
    mass = random(5,20);
    velocity = vel;
    acceleration = new PVector(0,0);
    grav = g;
    amp=0;
    scaledVel = vel;
  }

  void update(){
    position.add(scaledVel);
    velocity.add(acceleration);
    scaledVel = velocity.copy().mult(amp);
    
    
    if(scaledVel.mag() > maxVelo){
      scaledVel.normalize();
      scaledVel.mult(maxVelo);
    }
    
    acceleration.mult(0);
  }
  
  void display(){
    ellipse(position.x, position.y, mass, mass);
  }
  
  void applyForce(PVector force){
    PVector f = PVector.div(force,mass);
    acceleration.add(f);
  }
  
  void attract(Mover mover){
    PVector direction = PVector.sub(position, mover.position);
    float distance = constrain(direction.mag(), 20, 23);
    
    float strength = (grav * mass * mover.mass) / (distance * distance);
    PVector force = (direction.copy().normalize()).mult(strength);
    mover.applyForce(force);
  }
  
}
