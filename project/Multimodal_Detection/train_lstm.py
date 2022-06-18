import argparse, os, pickle
import numpy as np
from model_lstm import *
from dataset_lstm import *
from utils import AverageMeter, calc_loss, write_log, save_checkpoint, denorm, calc_accuracy
import re
import torch
from torch.utils import data
import torch.utils.data
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir', default='output', type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--train', default='', type=str)
parser.add_argument('--getthresholds', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--loss',default=2, type=int, help='the choice of losses')

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args;
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda;
    cuda = torch.device('cuda')

    model = Multimodal(img_dim=args.img_dim, network=args.net, num_layers_in_fc_layers=args.final_dim,
                      dropout=args.dropout)

    model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion;
    criterion = nn.CrossEntropyLoss()

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    least_loss = 0
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    global iteration;

    iteration = 0
    if args.train:
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading resumed checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
                args.start_epoch = checkpoint['epoch']
                iteration = checkpoint['iteration']
                model.load_state_dict(checkpoint['state_dict'])
                if not args.reset_lr:  
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("[Warning] no checkpoint found at '{}'".format(args.resume))

        if args.pretrain:
            if os.path.isfile(args.pretrain):
                print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
                checkpoint = torch.load(args.pretrain, map_location=torch.device('cuda'))
                model = neq_load_customized(model, checkpoint['state_dict'])
                print("=> loaded pretrained checkpoint '{}' (epoch {})"
                      .format(args.pretrain, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrain))

        transform = transforms.Compose([Scale(size=(args.img_dim, args.img_dim)), ToTensor(), Normalize()])
        train_loader = get_data(transform, 'train')

        global de_normalize;
        de_normalize = denorm()
        global img_path;
        img_path, model_path = set_path(args)
        global writer_train
        try:  # old version
            writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
        except:  # v1.7
            writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
        ### main loop ###
        for epoch in range(args.start_epoch, args.epochs):
            train_loss = train(train_loader, model, optimizer, epoch)

            # save curve
            writer_train.add_scalar('global/loss', train_loss, epoch)

            # save check_point
            # is_best = train_acc > best_acc; best_acc = max(train_acc, best_acc)
            print('Average loss of epoch[{epoch}] is {loss:.6f}'.format(epoch=epoch, loss=train_loss))
            save_checkpoint({'epoch': epoch + 1,
                             'net': args.net,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'iteration': iteration},
                            filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=False)

        print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    global num_epoch;
    global test_dissimilarity_score;
    global test_target;
    global test_number_ofile_number_of_chunks;
    global score_list;
    if args.getthresholds:
        if os.path.isfile(args.getthresholds):
            print("=> loading testing checkpoint '{}'".format(args.getthresholds))
            checkpoint = torch.load(args.getthresholds)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                sys.exit()
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.getthresholds, checkpoint['epoch']))
            num_epoch = checkpoint['epoch']
        elif args.getthresholds == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()
        transform = transforms.Compose([
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        train2_loader = get_data(transform, 'train')

        test_dissimilarity_score = {}

        test_target = {}

        test_number_ofile_number_of_chunks = {}
        score_list = {}
        test_loss = test(train2_loader, model)
        file_dissimilarity_score = open("file_dissimilarity_score_train.pkl", "wb")
        pickle.dump(test_dissimilarity_score, file_dissimilarity_score)
        file_dissimilarity_score.close()
        file_target = open("file_target_train.pkl", "wb")
        pickle.dump(test_target, file_target)
        file_target.close()
        file_number_of_chunks = open("file_number_of_chunks_train.pkl", "wb")
        pickle.dump(test_number_ofile_number_of_chunks, file_number_of_chunks)
        file_number_of_chunks.close()
        file_scorelist = open("file_scorelist.pkl", "wb")
        pickle.dump(score_list, file_scorelist)
        file_scorelist.close()
        sys.exit()
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                sys.exit()
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            # global num_epoch;
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        transform = transforms.Compose([
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        test_loader = get_data(transform, 'test')
        test_dissimilarity_score = {}
        test_target = {}
        test_number_ofile_number_of_chunks = {}
        score_list = {}
        test_loss = test(test_loader, model)
        file_dissimilarity_score = open("file_dissimilarity_score.pkl", "wb")
        pickle.dump(test_dissimilarity_score, file_dissimilarity_score)
        file_dissimilarity_score.close()
        file_target = open("file_target.pkl", "wb")
        pickle.dump(test_target, file_target)
        file_target.close()
        file_number_of_chunks = open("file_number_of_chunks.pkl", "wb")
        pickle.dump(test_number_ofile_number_of_chunks, file_number_of_chunks)
        file_number_of_chunks.close()
        file_scorelist = open("file_scorelist_test.pkl", "wb")
        pickle.dump(score_list, file_scorelist)
        file_scorelist.close()
        sys.exit()


def process_output(mask):
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.train()
    
    global iteration

    for idx, (video_seq, audio_seq, target,audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
        #print(audiopath[0].split('/')[-1])
        tic = time.time()

        #print(idx)
        #print(video_seq.shape)
        #print(audio_seq.shape)
        video_seq = video_seq.to(cuda)
        audio_seq = audio_seq.to(cuda)

        target = target.to(cuda)
        B = video_seq.size(0)
        vid_out = model.module.forward_vid(video_seq)
        aud_out = model.module.forward_aud(audio_seq)
        
        vid_class = model.module.final_classification_vid(vid_out)
        aud_class = model.module.final_classification_aud(aud_out)

        del video_seq
        del audio_seq

        loss1 = calc_loss(vid_out, aud_out, target, args.hyper_param)
        loss2 = criterion(vid_class, target.view(-1))
        loss3 = criterion(aud_class, target.view(-1))
        #print('L1',loss1.item())
        #print('L2',loss2.item())
        #print('L3',loss3.item())
        if args.loss == 1:
          loss = loss1 
        elif args.loss ==2:
          loss = loss1 + loss2
        else:
          loss = loss1 + loss2 + loss3
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  ' T:{3:.2f}\t'.format(
                epoch, idx, len(data_loader), time.time() - tic, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)

            iteration += 1

    return losses.local_avg


def get_data(transform, mode='test'):
    print('Loading data for "%s" ...' % mode)
    dataset = deepfake_3d(out_dir=args.out_dir, mode=mode,
                          transform=transform)

    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=32)
                                      
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def my_collate(batch):
    batch = list(filter(lambda x: x is not None and x[1].size()[3] == 99, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def test(data_loader, model):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (video_seq, audio_seq, target,audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
            #print(video_seq.shape)
            video_seq = video_seq.to(cuda)
            audio_seq = audio_seq.to(cuda)
            target = target.to(cuda)
            B = video_seq.size(0)

            vid_out = model.module.forward_vid(video_seq)
            aud_out = model.module.forward_aud(audio_seq)

            vid_class = model.module.final_classification_vid(vid_out)
            aud_class = model.module.final_classification_aud(aud_out)

            del video_seq
            del audio_seq

            loss1 = calc_loss(vid_out, aud_out, target, args.hyper_param)
            loss2 = criterion(vid_class, target.view(-1))
            loss3 = criterion(aud_class, target.view(-1))

            if args.loss == 1:
              loss = loss1 
            elif args.loss ==2:
              loss = loss1 + loss2
            else:
              loss = loss1 + loss2 + loss3
            losses.update(loss.item(), B)

            dist = torch.dist(vid_out[0, :].view(-1), aud_out[0, :].view(-1), 2)
            tar = target[0, :].view(-1).item()
            #loss = torch.mean(criterion(outputs, targets.type(torch.cuda.LongTensor)))
            #print(vid_class,tar)
            vid_name = audiopath[0].split('/')[-1]
            print(vid_name)
            if (test_dissimilarity_score.get(vid_name)):
                test_dissimilarity_score[vid_name] += dist
                test_number_ofile_number_of_chunks[vid_name] += 1
                score_list[vid_name].append(dist.item())
                # print(dist.item())
                # print(len(score_list[vid_name]))


            else:
                test_dissimilarity_score[vid_name] = dist
                test_number_ofile_number_of_chunks[vid_name] = 1
                score_list[vid_name] = []
                score_list[vid_name].append(dist.item())
                # print(dist.item())
            if (test_target.get(vid_name)):
                pass
            else:
                test_target[vid_name] = tar

    print('Loss {loss.avg:.4f}\t'.format(loss=losses))

    #write_log(content='Loss {loss.avg:.4f}\t'.format(loss=losses, args=args),
    #          epoch=num_epoch,
    #          filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
    return losses.avg


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = args.out_dir
    img_path = os.path.join(exp_path, 'img_lstm')
    model_path = os.path.join(exp_path, 'model_lstm')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()