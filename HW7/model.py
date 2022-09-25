import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf=64):
        super(Generator,self).__init__()
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        
        self.expand = nn.Sequential(
            nn.Linear(24, nc),
            nn.BatchNorm1d(nc),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolutio
            nn.ConvTranspose2d( nc + nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, c):
        z = z.view(-1,self.nz,1,1)
        c = self.expand(c).view(-1, self.nc, 1, 1)
        x = torch.cat((z, c), dim=1)
        return self.main(x)
        
        
class Discriminator(nn.Module):
    def __init__(self, img_shape, ndf=64):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.ndf = ndf
        
        self.expand = nn.Sequential(
            nn.Linear(24, self.img_shape[0] * self.img_shape[1]),
            nn.BatchNorm1d(self.img_shape[0] * self.img_shape[1]),
            nn.LeakyReLU()
        )
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, X, c):
        c = self.expand(c).view(-1, 1, self.img_shape[0], self.img_shape[1])   
        x = torch.cat((X, c), dim=1)
        return self.main(x).view(-1)        
        
        
        
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)